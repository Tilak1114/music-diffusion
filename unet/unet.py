import torch
import torch.nn as nn
from unet.embeddings import TimeStepEmbedding
from typing import Any, Dict, List, Optional, Tuple, Union
from unet.components import (
    get_down_block,
    UNetMidBlock2DCrossAttnMusic,
    get_up_block
)


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        block_out_channels = config['block_out_channels']
        down_block_types = config['down_block_types']
        up_block_types = config['up_block_types']

        self.in_channels = config['in_channels']
        self.pre_encoder_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.time_step_embedding = TimeStepEmbedding()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        cross_attention_dim = [1024] * len(down_block_types)
        attention_head_dim = config['attention_head_dim']

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=2,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=1280,
                add_downsample=not is_final_block,
                resnet_eps=1e-05,
                resnet_groups=32,
                cross_attention_dim=cross_attention_dim[i],
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=1,
                upcast_attention=True,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2DCrossAttnMusic(
            in_channels=block_out_channels[-1],
            temb_channels=1280,
            resnet_eps=1e-05,
            output_scale_factor=1,
            cross_attention_dim=cross_attention_dim[-1],
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=32,
            upcast_attention=True,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))

        self.num_upsamplers = 0

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(
                i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=2 + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=1280,
                add_upsample=add_upsample,
                resnet_eps=1e-05,
                resnet_groups=32,
                cross_attention_dim=reversed_cross_attention_dim[i],
                attn_num_head_channels=reversed_attention_head_dim[i],
                upcast_attention=True,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=32, eps=1e-05
        )
        self.conv_act = nn.SiLU()

        conv_out_padding = (3 - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], 8,
            kernel_size=3,
            padding=conv_out_padding
        )

    def forward(self,
                latents: torch.FloatTensor,
                timesteps,
                encoder_hidden_states: torch.Tensor,
                beat_features: torch.Tensor,
                chord_features: torch.Tensor,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                beat_attention_mask: Optional[torch.Tensor] = None,
                chord_attention_mask: Optional[torch.Tensor] = None,):

        timestep_embedding = self.time_step_embedding(timesteps)
        latents = self.pre_encoder_conv(latents)

        forward_upsample_size = True

        # ensure encoder_attention_mask is a bias, and make it broadcastable over multi-head-attention channels
        if encoder_attention_mask is not None:
            # if it's a mask: turn it into a bias. otherwise: assume it's already a bias
            if encoder_attention_mask.dtype is torch.bool:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(latents.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if beat_attention_mask is not None:  # Nic Modified
            # if it's a mask: turn it into a bias. otherwise: assume it's already a bias
            if beat_attention_mask.dtype is torch.bool:
                beat_attention_mask = (
                    1 - beat_attention_mask.to(latents.dtype)) * -10000.0
            beat_attention_mask = beat_attention_mask.unsqueeze(1)

        if chord_attention_mask is not None:  # Nic Modified
            # if it's a mask: turn it into a bias. otherwise: assume it's already a bias
            if chord_attention_mask.dtype is torch.bool:
                chord_attention_mask = (
                    1 - chord_attention_mask.to(latents.dtype)) * -10000.0
            chord_attention_mask = chord_attention_mask.unsqueeze(1)

        down_block_res_samples = (latents,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                latents, res_samples = downsample_block(
                    hidden_states=latents,
                    temb=timestep_embedding,
                    encoder_hidden_states=encoder_hidden_states,
                    beat_features=beat_features,
                    chord_features=chord_features,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    encoder_attention_mask=encoder_attention_mask,
                    beat_attention_mask=beat_attention_mask,
                    chord_attention_mask=chord_attention_mask
                )
            else:
                latents, res_samples = downsample_block(
                    hidden_states=latents,
                    temb=timestep_embedding
                )

            down_block_res_samples += res_samples

        latents = self.mid_block(
            latents, timestep_embedding,
            encoder_hidden_states=encoder_hidden_states,
            beat_features=beat_features,
            chord_features=chord_features,
            attention_mask=None,
            cross_attention_kwargs=None,
            encoder_attention_mask=encoder_attention_mask,
            beat_attention_mask=beat_attention_mask,
            chord_attention_mask=chord_attention_mask)

        # UP
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                latents = upsample_block(
                    hidden_states=latents,
                    temb=timestep_embedding,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    beat_features = beat_features,
                    chord_features = chord_features,
                    cross_attention_kwargs=None,
                    upsample_size=upsample_size,
                    attention_mask=None,
                    encoder_attention_mask=encoder_attention_mask,
                    beat_attention_mask = beat_attention_mask,
                    chord_attention_mask = chord_attention_mask
                )
            else:
                latents = upsample_block(
                    hidden_states=latents, 
                    temb=timestep_embedding, 
                    res_hidden_states_tuple=res_samples, 
                    upsample_size=upsample_size
                )
        
        latents = self.conv_act(
            self.conv_norm_out(latents)
            )
        
        return self.conv_out(latents)

