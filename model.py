import torch
import torch.nn as nn
import pytorch_lightning as pl
from model_utils import build_pretrained_models
import math
import soundfile as sf
import tqdm
import random
import torch.nn.functional as F
from unet.components import FundamentalMusicEmbedding, MusicPositionalEncoding, BeatEmbedding, ChordEmbedding
from tools import torch_tools
from transformers import get_scheduler
from sample_generation_helper import SampleGeneration
from diffusers.utils.torch_utils import randn_tensor

from transformers import AutoTokenizer
from transformers import T5EncoderModel
from unet import UNet
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from unet.components import BeatTokenizer, ChordTokenizer, BeatEmbedding, ChordEmbedding


class LatentMusicDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        pretrained_model_name = "audioldm-s-full"

        self.config = config
        self.vae, self.stft = build_pretrained_models(pretrained_model_name)
        self.vae.eval()
        self.stft.eval()

        self.model = MusicAudioDiffusion(
            config.pretrained_configs['text_encoder'],
            config.pretrained_configs['ddpm_scheduler'],
            config.module_config['unet'],
        )

        self.sample_gen = SampleGeneration(
            self.device,
            self.vae,
            self.stft,
            self.model
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.module_config['learning_rate'],
            betas=(
                self.config.module_config['adam_beta1'],
                self.config.module_config['adam_beta2']),
            weight_decay=self.config.module_config['weight_decay'],
            eps=self.config.module_config['adam_eps']
        )

        devices_in_use = self.trainer.num_devices
        effective_loader_size = len(
            self.trainer.datamodule.train_dataloader())/devices_in_use
        num_update_steps_per_epoch = math.ceil(
            effective_loader_size / 4)

        self.num_training_steps = self.config.trainer_config['max_epochs'] * \
            num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # or "epoch" based on your scheduler and training loop
                # How often to apply the scheduler. Usually, this is set to 1.
                "frequency": 1,
                # If your scheduler requires monitoring a value, specify it here. This field is optional and depends on your scheduler.
                "monitor": "validation_loss",
            }
        }

    def forward(self,
                true_latent,
                text, beats,
                chords, chords_time,
                validation_mode=False):
        return self.model(true_latent, text, beats, chords, chords_time, validation_mode=validation_mode)

    def training_step(self, batch, batch_idx):
        device = self.device
        target_length = int(10 * 102.4)

        text, audios, beats, chords, chords_time, _ = batch

        batch_size = len(audios)

        with torch.no_grad():
            mel, _, waveform = torch_tools.wav_to_fbank(
                audios,
                target_length,
                self.stft
            )
            # batch, 1, time, freq; [2, 1, 1024, 64]
            mel = mel.unsqueeze(1).to(device)
            # batch, channels, time_compressed, freq_compressed [2, 8, 256, 16]
            true_latent = self.vae.get_first_stage_encoding(
                self.vae.encode_first_stage(mel))

        loss = self.forward(true_latent, text, beats, chords,
                            chords_time, validation_mode=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        device = self.device
        text, audios, beats, chords, chords_time, _ = batch
        target_length = int(10 * 102.4)

        batch_size = len(audios)

        mel, _, waveform = torch_tools.wav_to_fbank(
            audios,
            target_length,
            self.stft
        )
        mel = mel.unsqueeze(1).to(device)
        true_latent = self.vae.get_first_stage_encoding(
            self.vae.encode_first_stage(mel)
        )
        val_loss = self.forward(
            true_latent,
            text,
            beats,
            chords,
            chords_time,
            validation_mode=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return val_loss

    def val_dataloader(self):
        return self.trainer.datamodule.val_dataloader()

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

    def on_train_epoch_end(self, unused=None):
        if self.trainer.global_rank == 0:
            # Get the current epoch
            current_epoch = self.current_epoch
            # Get the loss logged in the last training step
            last_loss = self.trainer.callback_metrics.get("train_loss")
            print(f"Epoch {current_epoch+1}, Train Loss: {last_loss}")

            val_dataloader = self.val_dataloader()

            prompt = next(iter(val_dataloader))[1]
            prompt = "This is an instrumental jam recording of a gear showcase. There is an electric guitar with a clear sound being played with an echo pedal. It gives the recording a dreamy feeling. This track can be used to lift guitar samples with effect for a beat. The chord sequence is D, G. The beat counts to 3. The bpm is 83.0. The key of this song is D minor."

            if self.current_epoch % 1 == 0:
                print(f"Generating for {prompt}")
                wav = self.sample_gen.generate(prompt)
                out = f"./tmp/output_{self.current_epoch}.wav"
                sf.write(out, wav, samplerate=16000)


class MusicAudioDiffusion(nn.Module):
    def __init__(
            self,
            text_encoder_name,
            scheduler_name,
            unet_config,
            d_fme=1024,  # FME
            fme_type="se",
            base=1,
            translation_bias_type="nd",
            emb_nn=True,
            d_pe=1024,  # PE
            if_index=True,
            if_global_timing=True,
            if_modulo_timing=False,
            d_beat=1024,  # Beat
            d_oh_beat_type=7,
            beat_len=50,
            d_chord=1024,  # Chord
            d_oh_chord_type=12,
            d_oh_inv_type=4,
            chord_len=20,
    ):
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_config = unet_config

        self.snr_gamma = 5

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.scheduler_name, subfolder="scheduler")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.text_encoder_name)

        self.unet = UNet(unet_config)

        # Music Feature Encoder
        self.FME = FundamentalMusicEmbedding(d_model=d_fme, base=base, if_trainable=False,
                                             type=fme_type, emb_nn=emb_nn, translation_bias_type=translation_bias_type)
        self.PE = MusicPositionalEncoding(
            d_model=d_pe, if_index=if_index, if_global_timing=if_global_timing, if_modulo_timing=if_modulo_timing)
        # self.PE2 = Music_PositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = if_modulo_timing, device = self.device)
        self.beat_tokenizer = BeatTokenizer(seq_len_beat=88, if_pad=True)
        self.beat_embedding_layer = BeatEmbedding(
            self.PE, d_model=d_beat, d_oh_beat_type=d_oh_beat_type)
        self.chord_embedding_layer = ChordEmbedding(
            self.FME, self.PE, d_model=d_chord, d_oh_type=d_oh_chord_type, d_oh_inv=d_oh_inv_type)
        self.chord_tokenizer = ChordTokenizer(
            seq_len_chord=chord_len, if_pad=True)

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(
            device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(
            device), batch.attention_mask.to(device)  # cuda

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(
            device)  # batch, len_text
        return encoder_hidden_states, boolean_encoder_mask

    def encode_beats(self, beats, device):
        # device = self.beat_embedding_layer.device
        out_beat = []
        out_beat_timing = []
        out_mask = []
        for beat in beats:
            tokenized_beats, tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer.tokenize(
                beat)
            out_beat.append(tokenized_beats)
            out_beat_timing.append(tokenized_beats_timing)
            out_mask.append(tokenized_beat_mask)
        out_beat, out_beat_timing, out_mask = torch.tensor(out_beat).to(device), torch.tensor(
            out_beat_timing).to(device), torch.tensor(out_mask).to(device)  # batch, len_beat
        embedded_beat = self.beat_embedding_layer(out_beat, out_beat_timing)

        return embedded_beat, out_mask

    def encode_chords(self, chords, chords_time, device):
        out_chord_root = []
        out_chord_type = []
        out_chord_inv = []
        out_chord_timing = []
        out_mask = []
        for chord, chord_time in zip(chords, chords_time):  # batch loop
            tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer.tokenize(
                chord, chord_time)
            out_chord_root.append(tokenized_chord_root)
            out_chord_type.append(tokenized_chord_type)
            out_chord_inv.append(tokenized_chord_inv)
            out_chord_timing.append(tokenized_chord_time)
            out_mask.append(tokenized_chord_mask)
        # chords: (B, LEN, 4)
        out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask = torch.tensor(out_chord_root).to(device), torch.tensor(
            out_chord_type).to(device), torch.tensor(out_chord_inv).to(device), torch.tensor(out_chord_timing).to(device), torch.tensor(out_mask).to(device)
        embedded_chord = self.chord_embedding_layer(
            out_chord_root, out_chord_type, out_chord_inv, out_chord_timing)
        return embedded_chord, out_mask
        # return out_chord_root, out_mask

    def forward(self, latents, prompt, beats, chords, chords_time, validation_mode=False):
        device = self.text_encoder.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)

        # with torch.no_grad():
        encoded_beats, beat_mask = self.encode_beats(
            beats, device)  # batch, len_beats, dim; batch, len_beats
        encoded_chords, chord_mask = self.encode_chords(
            chords, chords_time, device)

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * \
                torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)

        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)

        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            encoded_beats,
            encoded_chords,
            encoder_attention_mask=boolean_encoder_mask,
            beat_attention_mask=beat_mask,
            chord_attention_mask=chord_mask
        )

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(),
                              target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(),
                              target.float(), reduction="none")
            loss = loss.mean(
                dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def inference(self,
                  prompt,
                  beats,
                  chords,
                  chords_time,
                  inference_scheduler,
                  device,
                  num_steps=20,
                  num_samples_per_prompt=1,
                  disable_progress=True):
        device = self.text_encoder.device

        batch_size = len(prompt) * num_samples_per_prompt

        prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
        prompt_embeds = prompt_embeds.repeat_interleave(
            num_samples_per_prompt, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(
            num_samples_per_prompt, 0)

        encoded_beats, beat_mask = self.encode_beats(
            beats, device)  # batch, len_beats, dim; batch, len_beats
        encoded_beats = encoded_beats.repeat_interleave(
            num_samples_per_prompt, 0)
        beat_mask = beat_mask.repeat_interleave(num_samples_per_prompt, 0)

        encoded_chords, chord_mask = self.encode_chords(
            chords, chords_time, device)
        encoded_chords = encoded_chords.repeat_interleave(
            num_samples_per_prompt, 0)
        chord_mask = chord_mask.repeat_interleave(num_samples_per_prompt, 0)

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

        num_warmup_steps = len(timesteps) - num_steps * \
            inference_scheduler.order

        for i, t in enumerate(timesteps):

            latent_model_input = inference_scheduler.scale_model_input(
                latents, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask,
                beat_features=encoded_beats, beat_attention_mask=beat_mask, chord_features=encoded_chords, chord_attention_mask=chord_mask
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(
                noise_pred, t, latents).prev_sample
        
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = randn_tensor(shape, generator=None,
                               device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents
