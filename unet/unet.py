import torch
import torch.nn as nn
from unet.embeddings import TimeStepEmbedding, FloatFeatureEmbedding
from unet.components import (
    get_down_block,
    UNetMidBlock3DCrossAttnMusic,
    get_up_block
)

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        block_out_channels = config['block_out_channels']
        down_block_types = config['down_block_types']
        up_block_types = config['up_block_types']

        use_prompt_embedding = config['use_text_conditioning']

        self.in_channels = config['in_channels']
        self.pre_encoder_conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=block_out_channels[0],
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1
        )
        self.time_step_embedding = TimeStepEmbedding()

        self.rbg_mean_embedding = FloatFeatureEmbedding(1, 256)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        vid_cross_attention_dim = [512] * len(down_block_types)
        tempo_cross_attention_dim = [256] * len(down_block_types)
        prompt_cross_attention_dim = [1024] * len(down_block_types)
        
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
                vid_cross_attention_dim=vid_cross_attention_dim[i],
                prompt_cross_attention_dim=prompt_cross_attention_dim[i],
                tempo_cross_attention_dim=tempo_cross_attention_dim[i],
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3DCrossAttnMusic(
            in_channels=block_out_channels[-1],
            temb_channels=1280,
            resnet_eps=1e-05,
            cross_attention_dims=[vid_cross_attention_dim[-1], 
                                  tempo_cross_attention_dim[-1],
                                  prompt_cross_attention_dim[-1] 
                                  ],
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=32,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_vid_cross_attention_dim = list(reversed(vid_cross_attention_dim))
        reversed_tempo_cross_attention_dim = list(reversed(tempo_cross_attention_dim))
        reversed_prompt_cross_attention_dim = list(reversed(prompt_cross_attention_dim))

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
                attn_num_head_channels=reversed_attention_head_dim[i],
                vid_cross_attention_dim=reversed_vid_cross_attention_dim[i],
                prompt_cross_attention_dim=reversed_prompt_cross_attention_dim[i],
                tempo_cross_attention_dim=reversed_tempo_cross_attention_dim[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=32, eps=1e-05
        )
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv3d(
            block_out_channels[0], 8,
            kernel_size=(3, 3, 3),
            padding=1
        )

    def forward(self,
                latents: torch.FloatTensor,
                timesteps,
                video_emb = None,
                rgb_mean = None,
                prompt_emb = None,
                ):

        timestep_embedding = self.time_step_embedding(timesteps)
        latents = self.pre_encoder_conv(latents)

        if rgb_mean is not None:
            rgb_mean = rgb_mean.to(timestep_embedding.device)
            rgb_embedding = self.rbg_mean_embedding(rgb_mean)
        else:
            rgb_embedding = None

        forward_upsample_size = True

        down_block_res_samples = (latents,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                latents, res_samples = downsample_block(
                    hidden_states=latents,
                    temb=timestep_embedding,
                    prompt_emb=prompt_emb,
                    vid_emb=video_emb,
                    tempo_emb=rgb_embedding,
                )
            else:
                latents, res_samples = downsample_block(
                    hidden_states=latents,
                    temb=timestep_embedding
                )

            down_block_res_samples += res_samples

        latents = self.mid_block(
            latents, 
            timestep_embedding,
            )

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
                    upsample_size=upsample_size,
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

