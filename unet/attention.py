import math
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        # 1. Self-Attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            cross_attention_dim=None,
        )

        self.ff = FeedForward(dim, dropout=dropout, final_dropout=False)

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=False,
                cross_attention_dim=cross_attention_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        
        norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
        )
        
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None and encoder_hidden_states is not None:
            norm_hidden_states = (
               self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            # print (f"attn output {attn_output.shape}, hidden states {hidden_states.shape}")
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states

class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fn = GEGLU(dim, inner_dim)
        
        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim, bias=True))
        self.to_out.append(nn.Dropout(dropout))

    def forward(self, 
                hidden_states, 
                encoder_hidden_states=None,):
        
        batch_size = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
       
        query = query.float()
        key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None):
        if batch_size is None:
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length > target_length:
            # we *could* trim the mask with:
            #   attention_mask = attention_mask[:,:target_length]
            # but this is weird enough that it's more likely to be a mistake than a shortcut
            raise ValueError(f"mask's length ({current_length}) exceeds the sequence length ({target_length}).")
        elif current_length < target_length:
            remaining_length: int = target_length - current_length
            attention_mask = F.pad(attention_mask, (0, remaining_length), value=0.0)
                

        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask