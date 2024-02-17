import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FundamentalMusicEmbedding(nn.Module):
    def __init__(self, d_model, base, if_trainable=False, if_translation_bias_trainable=True, type="se", emb_nn=None, translation_bias_type="nd"):
        super().__init__()
        self.d_model = d_model
        self.base = base
        self.if_trainable = if_trainable  # whether the se is trainable

        if translation_bias_type is not None:
            self.if_translation_bias = True
            # default the 2d vector is trainable
            self.if_translation_bias_trainable = if_translation_bias_trainable
            if translation_bias_type == "2d":
                # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
                translation_bias = torch.rand((1, 2), dtype=torch.float32)
            elif translation_bias_type == "nd":
                translation_bias = torch.rand(
                    (1, self.d_model), dtype=torch.float32)
            translation_bias = nn.Parameter(
                translation_bias, requires_grad=True)
            self.register_parameter("translation_bias", translation_bias)
        else:
            self.if_translation_bias = False

        i = torch.arange(d_model)
        angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / d_model)
        angle_rates = angle_rates[None, ...]

        if self.if_trainable:
            angles = nn.Parameter(angle_rates, requires_grad=True)
            self.register_parameter("angles", angles)

        else:
            self.angles = angle_rates

    def forward(self, inp):
        if inp.dim() == 2:
            inp = inp[..., None]  # pos (batch, num_pitch, 1)
        elif inp.dim() == 1:
            inp = inp[None, ..., None]  # pos (1, num_pitch, 1)
        # (batch, num_pitch)*(1,dim)
        angle_rads = inp*self.angles.to(inp.device)

        # apply sin to even indices in the array; 2i
        angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, :, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

        pos_encoding = angle_rads.to(torch.float32)
        if self.if_translation_bias:
            if self.translation_bias.size()[-1] != self.d_model:
                translation_bias = self.translation_bias.repeat(
                    1, 1, int(self.d_model/2))
            else:
                translation_bias = self.translation_bias
            pos_encoding += translation_bias
        else:
            self.translation_bias = None
        return pos_encoding


class MusicPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, if_index=True,
                 if_global_timing=True, if_modulo_timing=True):
        super().__init__()
        self.if_index = if_index
        self.if_global_timing = if_global_timing
        self.if_modulo_timing = if_modulo_timing
        self.dropout = nn.Dropout(p=dropout)
        self.index_embedding = FundamentalMusicEmbedding(
            d_model=d_model, base=10000, if_trainable=False, translation_bias_type=None, if_translation_bias_trainable=False, type="se")
        self.global_time_embedding = FundamentalMusicEmbedding(
            d_model=d_model, base=10001, if_trainable=False, translation_bias_type=None, if_translation_bias_trainable=False, type="se")
        self.modulo_time_embedding = FundamentalMusicEmbedding(
            d_model=d_model, base=10001, if_trainable=False, translation_bias_type=None, if_translation_bias_trainable=False, type="se")

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        '''
		if self.if_global_timing:
			print("pe add global time")
		if self.if_modulo_timing:
			print("pe add modulo time")
		if self.if_index:
			print("pe add idx")
		'''

    def forward(self, inp, dur_onset_cumsum=None):

        if self.if_index:
            # [seq_len, batch_size, embedding_dim]
            pe_index = self.pe[:inp.size(1)]
            # [batch_size, seq_len, embedding_dim]
            pe_index = torch.swapaxes(pe_index, 0, 1)
            inp += pe_index

        if self.if_global_timing:
            global_timing = dur_onset_cumsum
            global_timing_embedding = self.global_time_embedding(global_timing)
            inp += global_timing_embedding

        if self.if_modulo_timing:
            modulo_timing = dur_onset_cumsum % 4
            modulo_timing_embedding = self.modulo_time_embedding(modulo_timing)
            inp += modulo_timing_embedding
        return self.dropout(inp)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[:x.size(1)]  # [seq_len, batch_size, embedding_dim]
        pos = torch.swapaxes(pos, 0, 1)  # [batch_size, seq_len, embedding_dim]
        print("huh????", pos.shape, x.shape)
        x = x + pos
        return self.dropout(x)


class ChordTokenizer():
    def __init__(self, seq_len_chord=88, if_pad=True):
        self.pitch_dict = {'pad': 0, "None": 1, "N": 1, "A": 2, "A#": 3, "Bb": 3, "B": 4, "Cb": 4, "B#": 5, "C": 5, "C#": 6,
                           "Db": 6, "D": 7, "D#": 8, "Eb": 8, "E": 9, "Fb": 9, "E#": 10, "F": 10, "F#": 11, "Gb": 11, "G": 12, "G#": 13, "Ab": 13}
        self.chord_type_dict = {'pad': 0, "None": 1, "N": 1, "maj": 2, "maj7": 3, "m": 4,
                                "m6": 5, "m7": 6, "m7b5": 7, "6": 8, "7": 9, "aug": 10, "dim": 11}  # , "/":
        self.chord_inversion_dict = {
            'pad': 0, "None": 1, "N": 1, "inv": 2, "no_inv": 3}
        self.seq_len_chord = seq_len_chord
        self.if_pad = if_pad

    def tokenize(self, chord, chord_time):

        if len(chord) == 0:
            chord, chord_time = ["N"], [0.]

        if self.if_pad:
            pad_len_chord = self.seq_len_chord - len(chord)
            chord_mask = [True]*len(chord) + [False]*pad_len_chord

            chord += ["pad"]*pad_len_chord
            chord_time += [chord_time[-1]]*pad_len_chord

        else:
            chord_mask = [True]*len(chord)

        self.chord_root, self.chord_type, self.chord_inv = self.tokenize_chord_lst(
            chord)
        self.chord_time = chord_time
        self.chord_mask = chord_mask

        return self.chord_root, self.chord_type, self.chord_inv, self.chord_time, self.chord_mask

    def get_chord_root_type_inversion_timestamp(self, chord):
        if chord == "pad":
            return "pad", "pad", "pad"

        if chord == "N":
            return "N", "N", "N"

        if len(chord.split('/')) > 1:
            chord_inv = "inv"
        else:
            chord_inv = "no_inv"

        chord_wo_inv = chord.split('/')[0]

        if len(chord_wo_inv) > 1:  # this part might have a '#' or 'b'
            if chord_wo_inv[1] == '#' or chord_wo_inv[1] == 'b':
                chord_root = chord_wo_inv[0:2]
            else:
                chord_root = chord_wo_inv[0]
        else:
            chord_root = chord_wo_inv[0]

        if len(chord_wo_inv) > len(chord_root):
            chord_type = chord_wo_inv[len(chord_root):]
        else:
            chord_type = 'maj'

        return chord_root, chord_type, chord_inv

    def tokenize_chord_lst(self, chord_lst):
        out_root = []
        out_type = []
        out_inv = []
        for chord in chord_lst:
            chord_root, chord_type, chord_inversion = self.get_chord_root_type_inversion_timestamp(
                chord)
            out_root.append(self.pitch_dict[chord_root])
            out_type.append(self.chord_type_dict[chord_type])
            out_inv.append(self.chord_inversion_dict[chord_inversion])
        return out_root, out_type, out_inv


class BeatTokenizer():
    def __init__(self, seq_len_beat=88, if_pad=True):
        self.beat_dict = {'pad': 0, "None": 1, 1.: 2,
                          2.: 3, 3.: 4, 4.: 5, 5.: 6, 6.: 7, 7.: 8}
        self.if_pad = if_pad
        self.seq_len_beat = seq_len_beat

    def tokenize(self, beat_lst):
        # beats = [[0.56, 1.1, 1.66, 2.24, 2.8, 3.36, 3.92, 4.48, 5.04, 5.6, 6.16, 6.74, 7.32, 7.9, 8.46, 9.0, 9.58], [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]]
        if self.if_pad:
            if len(beat_lst[0]) == 0:
                beat_mask = [False]*self.seq_len_beat
                beat_lst = [[0.]*self.seq_len_beat, ["pad"]*self.seq_len_beat]
            else:
                pad_len_beat = self.seq_len_beat - len(beat_lst[0])
                beat_mask = [True]*len(beat_lst[0]) + [False]*pad_len_beat
                beat_lst = [beat_lst[0]+[beat_lst[0][-1]] *
                            pad_len_beat,  beat_lst[1]+["pad"]*pad_len_beat]

        else:
            beat_mask = [True]*len(beat_lst[0])
        self.beat = [self.beat_dict[x] for x in beat_lst[1]]
        self.beat_timing = beat_lst[0]

        return self.beat, self.beat_timing, beat_mask


class ChordEmbedding(nn.Module):
    def __init__(self, FME, PE, d_model=256,
                 d_oh_type=12, d_oh_inv=4):
        super().__init__()
        self.FME = FME
        self.PE = PE
        self.d_model = d_model
        self.d_oh_type = d_oh_type
        self.d_oh_inv = d_oh_inv
        self.chord_ffn = nn.Linear(
            d_oh_type + d_oh_inv + d_model + d_model, d_model)

    def forward(self, chord_root, chord_type, chord_inv, chord_timing):
        chord_root_emb = self.FME(chord_root)
        chord_type_emb = F.one_hot(chord_type.to(
            torch.int64), num_classes=self.d_oh_type).to(torch.float32)
        chord_inv_emb = F.one_hot(chord_inv.to(
            torch.int64), num_classes=self.d_oh_inv).to(torch.float32)
        chord_time_emb = self.PE.global_time_embedding(chord_timing)

        chord_emb = self.chord_ffn(torch.cat(
            (chord_root_emb, chord_type_emb, chord_inv_emb, chord_time_emb), dim=-1))

        return chord_emb


class BeatEmbedding(nn.Module):
    def __init__(self, PE, d_model=256, d_oh_beat_type=4):
        super().__init__()
        self.PE = PE
        self.d_model = d_model
        self.d_oh_beat_type = d_oh_beat_type
        self.beat_ffn = nn.Linear(d_oh_beat_type+d_model, d_model)

    def forward(self, beats, beats_timing):

        beat_type_emb = F.one_hot(
            beats.to(torch.int64), num_classes=self.d_oh_beat_type).to(torch.float32)
        beat_time_emb = self.PE.global_time_embedding(beats_timing)
        merged_beat = torch.cat((beat_type_emb, beat_time_emb), dim=-1)

        beat_emb = self.beat_ffn(merged_beat)
        return beat_emb


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    if len(timesteps.shape) == 0:
        timesteps = timesteps.unsqueeze(0)
        
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimeStepEmbedding(nn.Module):
    def __init__(self):
        super(TimeStepEmbedding, self).__init__()
        self.linear1 = nn.Linear(320, 1280)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(1280, 1280)

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            embedding_dim=320,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        x = self.linear2(
            self.activation(
                self.linear1(t_emb)
            )
        )

        return x


# class CrossAttnDownBlock2DMusic(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         temb_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         attn_num_head_channels=1,
#         cross_attention_dim=1280,
#         output_scale_factor=1.0,
#         downsample_padding=1,
#         add_downsample=True,
#         use_linear_projection=False,
#         only_cross_attention=False,
#         upcast_attention=False,
#     ):
#         super().__init__()
#         resnets = []
#         attentions = []
#         attentions2 = []
#         attentions3 = []

#         self.has_cross_attention = True
#         self.attn_num_head_channels = attn_num_head_channels

#         for i in range(num_layers):
#             in_channels = in_channels if i == 0 else out_channels
#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
#             )

#             attentions.append(
#                 Transformer2DModel(
#                     attn_num_head_channels,
#                     out_channels // attn_num_head_channels,
#                     in_channels=out_channels,
#                     num_layers=1,
#                     cross_attention_dim=cross_attention_dim,
#                     norm_num_groups=resnet_groups,
#                     use_linear_projection=use_linear_projection,
#                     only_cross_attention=only_cross_attention,
#                     upcast_attention=upcast_attention,
#                 ))
#             attentions2.append(
#                 Transformer2DModel(
#                     attn_num_head_channels,
#                     out_channels // attn_num_head_channels,
#                     in_channels=out_channels,
#                     num_layers=1,
#                     cross_attention_dim=cross_attention_dim,
#                     norm_num_groups=resnet_groups,
#                     use_linear_projection=use_linear_projection,
#                     only_cross_attention=only_cross_attention,
#                     upcast_attention=upcast_attention,
#                 ))
#             attentions3.append(
#                 Transformer2DModel(
#                     attn_num_head_channels,
#                     out_channels // attn_num_head_channels,
#                     in_channels=out_channels,
#                     num_layers=1,
#                     cross_attention_dim=cross_attention_dim,
#                     norm_num_groups=resnet_groups,
#                     use_linear_projection=use_linear_projection,
#                     only_cross_attention=only_cross_attention,
#                     upcast_attention=upcast_attention,
#                 )
#             )
                

#         self.attentions = nn.ModuleList(attentions)
#         self.attentions2 = nn.ModuleList(attentions2)
#         self.attentions3 = nn.ModuleList(attentions3)
#         self.resnets = nn.ModuleList(resnets)

#         if add_downsample:
#             self.downsamplers = nn.ModuleList(
#                 [
#                     Downsample2D(
#                         out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
#                     )
#                 ]
#             )
#         else:
#             self.downsamplers = None

#         self.gradient_checkpointing = False

#     def forward(
#         self,
#         hidden_states: torch.FloatTensor,
#         temb: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         beat_features = None,
#         chord_features = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         beat_attention_mask = None,
#         chord_attention_mask = None
#     ):
#         output_states = ()

#         for resnet, attn, attn2, attn3 in zip(self.resnets, self.attentions, self.attentions2, self.attentions3):
#             hidden_states = resnet(hidden_states, temb)
#             hidden_states = attn(
#                 hidden_states,
#                 encoder_hidden_states=encoder_hidden_states,
#                 cross_attention_kwargs=cross_attention_kwargs,
#                 attention_mask=attention_mask,
#                 encoder_attention_mask=encoder_attention_mask,
#             ).sample

#             hidden_states = attn2(
#                 hidden_states,
#                 encoder_hidden_states=beat_features,
#                 cross_attention_kwargs=cross_attention_kwargs,
#                 attention_mask=attention_mask,
#                 encoder_attention_mask=beat_attention_mask,
#             ).sample

#             hidden_states = attn3(
#                 hidden_states,
#                 encoder_hidden_states=chord_features,
#                 cross_attention_kwargs=cross_attention_kwargs,
#                 attention_mask=attention_mask,
#                 encoder_attention_mask=chord_attention_mask,
#             ).sample

#             output_states += (hidden_states,)

#         if self.downsamplers is not None:
#             for downsampler in self.downsamplers:
#                 hidden_states = downsampler(hidden_states)

#             output_states += (hidden_states,)

#         return hidden_states, output_states
    
# class ResnetBlock2D(nn.Module):
#     def __init__(
#         self,
#         *,
#         in_channels,
#         out_channels=None,
#         conv_shortcut=False,
#         dropout=0.0,
#         temb_channels=512,
#         groups=32,
#         groups_out=None,
#         pre_norm=True,
#         eps=1e-6,
#         kernel=None,
#         output_scale_factor=1.0,
#         use_in_shortcut=None,
#         up=False,
#         down=False,
#         conv_shortcut_bias: bool = True,
#         conv_2d_out_channels: Optional[int] = None,
#     ):
#         super().__init__()
#         self.pre_norm = pre_norm
#         self.pre_norm = True
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut
#         self.up = up
#         self.down = down
#         self.output_scale_factor = output_scale_factor

#         if groups_out is None:
#             groups_out = groups

#         self.norm1 = torch.nn.GroupNorm(
#             num_groups=groups, 
#             num_channels=in_channels, 
#             eps=eps, 
#             affine=True) 

#         self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#         self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        
#         self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
            
#         self.dropout = torch.nn.Dropout(dropout)
#         conv_2d_out_channels = conv_2d_out_channels or out_channels
#         self.conv2 = torch.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

#         self.nonlinearity = nn.SiLU()

#         self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

#         self.conv_shortcut = None
#         if self.use_in_shortcut:
#             self.conv_shortcut = torch.nn.Conv2d(
#                 in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
#             )

#     def forward(self, input_tensor, temb):
#         hidden_states = input_tensor

#         hidden_states = self.norm1(hidden_states) 

#         hidden_states = self.nonlinearity(hidden_states)

#         hidden_states = self.conv1(hidden_states)

#         temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
#         hidden_states = hidden_states + temb

#         hidden_states = self.nonlinearity(self.norm2(hidden_states))

#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.conv2(hidden_states)

#         if self.conv_shortcut is not None:
#             input_tensor = self.conv_shortcut(input_tensor)

#         output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

#         return output_tensor
