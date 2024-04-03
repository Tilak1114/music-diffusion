import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from unet.attention import BasicTransformerBlock


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    attn_num_head_channels,
    resnet_groups=None,
    vid_cross_attention_dim=None,
    tempo_cross_attention_dim=None,
    prompt_cross_attention_dim=None,
    downsample_padding=None,
):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
        )
    elif down_block_type == "CrossAttnDownBlock3DMusic":
        return CrossAttnDownBlock3DMusic(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            cross_attention_dims=[vid_cross_attention_dim,
                                  tempo_cross_attention_dim,
                                  prompt_cross_attention_dim]
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    attn_num_head_channels,
    resnet_groups=None,
    vid_cross_attention_dim=None,
    tempo_cross_attention_dim=None,
    prompt_cross_attention_dim=None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith(
        "UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
        )

    elif up_block_type == "CrossAttnUpBlock3DMusic":

        return CrossAttnUpBlock3DMusic(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            attn_num_head_channels=attn_num_head_channels,
            cross_attention_dims=[vid_cross_attention_dim,
                                  tempo_cross_attention_dim,
                                  prompt_cross_attention_dim]
        )

    raise ValueError(f"{up_block_type} does not exist.")


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock3DCrossAttnMusic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        attn_num_head_channels=1,
        cross_attention_dims=[]
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        layerwise_attention_list = []

        resnet_groups = resnet_groups if resnet_groups is not None else min(
            in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
            )
        ]

        for _ in range(num_layers):
            attention_list = []
            for j in range(len(cross_attention_dims)):
                attention_list.append(
                    Transformer3DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dims[j],
                        norm_num_groups=resnet_groups,
                    )
                )
                
            layerwise_attention_list.append(attention_list)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.layerwise_attention_list = layerwise_attention_list

        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        vid_emb: Optional[torch.FloatTensor] = None,
        tempo_emb: Optional[torch.FloatTensor] = None,
        prompt_emb=None
    ) -> torch.FloatTensor:
        embs = [vid_emb, tempo_emb, prompt_emb]
        hidden_states = self.resnets[0](hidden_states, temb)
        for i, resnet in enumerate(self.resnets[1:]):
            for j, attn in enumerate(self.layerwise_attention_list[i]):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=embs[j],
                )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class Transformer3DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Input normalization and projection are adapted for 3D
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=norm_elementwise_affine)

        # Replace the Linear layers with Conv3d to handle the temporal dimension.
        # Kernel size (1, 1, 1) and stride (1, 1, 1) to maintain spatial and temporal dimensions.
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_channels = in_channels if out_channels is None else out_channels
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ):
        # Input processing adapts for the added temporal dimension
        batch, _, depth, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]

        # Reshape and permute operations adapted for 3D (including temporal dimension)
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch, depth * height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer blocks process the reshaped hidden states
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch, depth, height, width, inner_dim).permute(0, 4, 1, 2, 3).contiguous()
        output = hidden_states + residual

        return output


class CrossAttnDownBlock3DMusic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        attn_num_head_channels=1,
        cross_attention_dims=[],
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        layer_wise_attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )
            
            attention_list = []

            for j in range(len(cross_attention_dims)):
                attention_list.append(
                    Transformer3DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dims[j],
                        norm_num_groups=resnet_groups,
                    )
                )

            layer_wise_attentions.append(attention_list)

        self.layer_wise_attentions = layer_wise_attentions

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        vid_emb: Optional[torch.FloatTensor] = None,
        tempo_emb: Optional[torch.FloatTensor] = None,
        prompt_emb=None
    ):
        output_states = ()

        attn_embs = [vid_emb, tempo_emb, prompt_emb]

        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)

            for j, attn in enumerate(self.layer_wise_attentions[i]):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=attn_embs[j],
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class Downsample3D(nn.Module):
    """
    A downsampling layer with an optional convolution for 3D inputs.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels: output channels, if different from input channels.
        padding: padding size for the convolutional layer.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=(1, 1, 1),):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = (2, 2, 2)

        if use_conv:
            self.conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=(
                3, 3, 3), stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels, "Channels must match for AvgPool3d when use_conv is False"
            self.conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1, 0, 1)  # Padding for depth, height, and width
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat(
                [hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        eps=1e-6,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True)

        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = torch.nn.Conv3d(
            out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv3d(
                in_channels, conv_3d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.time_emb_proj(self.nonlinearity(temb))[
            :, :, None, None, None]
        hidden_states = hidden_states + temb

        hidden_states = self.nonlinearity(self.norm2(hidden_states))

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states)

        return output_tensor


class CrossAttnUpBlock3DMusic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        attn_num_head_channels=1,
        add_upsample=True,
        cross_attention_dims=[],
    ):
        super().__init__()
        resnets = []
        layer_wise_attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

            attention_list = []

            for j in range(len(cross_attention_dims)):
                attention_list.append(
                    Transformer3DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dims[j],
                        norm_num_groups=resnet_groups,
                    )
                )
            
            layer_wise_attentions.append(attention_list)

        self.layer_wise_attentions = layer_wise_attentions

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        prompt_emb=None,
        video_emb=None,
        tempo_emb=None,
        upsample_size: Optional[int] = None,
    ):
        attn_embs = [video_emb, tempo_emb, prompt_emb]

        for i, resnet in enumerate(self.resnets):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat(
                [hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

            for j, attn in enumerate(self.layer_wise_attentions[i]):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=attn_embs[j],
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states,
                                          upsample_size)

        return hidden_states


class Upsample3D(nn.Module):
    """
    A 3D upsampling layer with an optional 3D convolution.

    Parameters:
        channels: Number of channels in the input and output.
        use_conv: If True, use a convolution operation after upsampling.
        use_conv_transpose: If True, use a transposed convolution for upsampling.
        out_channels: Number of output channels.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            # Adjust kernel size, stride, and padding for 3D.
            self.conv = nn.ConvTranspose3d(channels, self.out_channels, kernel_size=(
                2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1))
        elif use_conv:
            # Use a 3D convolution with appropriate padding.
            self.conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        else:
            self.conv = None

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            # If using conv transpose, directly return its output.
            return self.conv(hidden_states)
        else:
            # Use torch.nn.functional.interpolate for 3D upsampling.
            if output_size is None:
                # Adjust scale_factor for 3D (temporal, height, width).
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
            else:
                # If output_size is specified, use it to set the output size.
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest")

            if self.use_conv:
                # Apply the convolution if specified.
                hidden_states = self.conv(hidden_states)

        return hidden_states
