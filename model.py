import torch
import torch.nn as nn
import pytorch_lightning as pl
from model_utils import build_pretrained_models
import math
import soundfile as sf
import tqdm
import random
from components import FundamentalMusicEmbedding, MusicPositionalEncoding, BeatEmbedding, ChordEmbedding
from tools import torch_tools
from transformers import SchedulerType, get_scheduler
from sample_generation_helper import SampleGeneration

from transformers import AutoTokenizer
from transformers import T5EncoderModel


class LatentMusicDiffusionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        pretrained_model_name = "audioldm2-music"
        self.args = args
        self.vae, self.stft = build_pretrained_models(pretrained_model_name)
        self.vae.eval()
        self.stft.eval()

        self.model = MusicAudioDiffusion(
            args.text_encoder_name, 
            args.scheduler_name, 
            args.unet_model_name, 
            args.unet_model_config, 
            args.snr_gamma, 
            args.freeze_text_encoder, 
            args.uncondition 
        )

        self.sample_gen = SampleGeneration(
            self.vae, 
            self.stft, self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        devices_in_use = self.trainer.num_devices
        effective_loader_size = len(self.trainer.datamodule.train_dataloader())/devices_in_use
        num_update_steps_per_epoch = math.ceil(effective_loader_size / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

        # Define the learning rate scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # or "epoch" based on your scheduler and training loop
                "frequency": 1,  # How often to apply the scheduler. Usually, this is set to 1.
                "monitor": "validation_loss",  # If your scheduler requires monitoring a value, specify it here. This field is optional and depends on your scheduler.
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
            mel = mel.unsqueeze(1).to(device) #batch, 1, time, freq; [2, 1, 1024, 64]
            true_latent = self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel)) #batch, channels, time_compressed, freq_compressed [2, 8, 256, 16]

        loss = self.forward(true_latent, text, beats, chords, chords_time, validation_mode=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
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
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
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
		unet_model_config_path=None,
		snr_gamma=None,
		freeze_text_encoder=True,
		uncondition=False,
		d_fme = 1024,  #FME
		fme_type = "se", 
		base = 1, 
		translation_bias_type = "nd",
		emb_nn = True,
		d_pe = 1024, #PE
		if_index = True, 
		if_global_timing = True,
		if_modulo_timing = False,
		d_beat = 1024, #Beat
		d_oh_beat_type = 7, 
		beat_len = 50,
		d_chord = 1024, #Chord
		d_oh_chord_type = 12,
		d_oh_inv_type = 4,
		chord_len = 20,
	):
		super().__init__()

		assert unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

		self.text_encoder_name = text_encoder_name
		self.scheduler_name = scheduler_name
		self.unet_model_config_path = unet_model_config_path
		self.snr_gamma = snr_gamma
		self.freeze_text_encoder = freeze_text_encoder
		self.uncondition = uncondition

		# https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
		self.scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
		self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
		self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)

		unet_config = UNet2DConditionModelMusic.load_config(unet_model_config_path)
		self.unet = UNet2DConditionModelMusic.from_config(unet_config, subfolder="unet")

		#Music Feature Encoder
		self.FME = FundamentalMusicEmbedding(d_model = d_fme, base= base, if_trainable = False, type = fme_type,emb_nn=emb_nn,translation_bias_type = translation_bias_type)
		self.PE = MusicPositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = if_modulo_timing)
		# self.PE2 = Music_PositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = if_modulo_timing, device = self.device)
		self.beat_tokenizer = beat_tokenizer(seq_len_beat=beat_len, if_pad = True)
		self.beat_embedding_layer = BeatEmbedding(self.PE, d_model = d_beat, d_oh_beat_type = d_oh_beat_type)
		self.chord_embedding_layer = ChordEmbedding(self.FME, self.PE, d_model = d_chord, d_oh_type = d_oh_chord_type, d_oh_inv = d_oh_inv_type)
		self.chord_tokenizer = chord_tokenizer(seq_len_chord=chord_len, if_pad = True)


	def compute_snr(self, timesteps):
		"""
		Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
		"""
		alphas_cumprod = self.noise_scheduler.alphas_cumprod
		sqrt_alphas_cumprod = alphas_cumprod**0.5
		sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

		# Expand the tensors.
		# Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
		sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
		while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
			sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
		alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

		sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
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
		input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device) #cuda
		if self.freeze_text_encoder:
			with torch.no_grad():
				encoder_hidden_states = self.text_encoder(
					input_ids=input_ids, attention_mask=attention_mask
				)[0] #batch, len_text, dim
		else:
			encoder_hidden_states = self.text_encoder(
				input_ids=input_ids, attention_mask=attention_mask
			)[0]
		boolean_encoder_mask = (attention_mask == 1).to(device) ##batch, len_text
		return encoder_hidden_states, boolean_encoder_mask

	def encode_beats(self, beats): 
		# device = self.beat_embedding_layer.device
		out_beat = []
		out_beat_timing = []
		out_mask = []
		for beat in beats:
			tokenized_beats,tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
			out_beat.append(tokenized_beats)
			out_beat_timing.append(tokenized_beats_timing)
			out_mask.append(tokenized_beat_mask)
		out_beat, out_beat_timing, out_mask = torch.tensor(out_beat).cuda(), torch.tensor(out_beat_timing).cuda(), torch.tensor(out_mask).cuda() #batch, len_beat
		embedded_beat = self.beat_embedding_layer(out_beat, out_beat_timing)

		return embedded_beat, out_mask

	def encode_chords(self, chords,chords_time):
		out_chord_root = []
		out_chord_type = []
		out_chord_inv = []
		out_chord_timing = []
		out_mask = []
		for chord, chord_time in zip(chords,chords_time): #batch loop
			tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
			out_chord_root.append(tokenized_chord_root)
			out_chord_type.append(tokenized_chord_type)
			out_chord_inv.append(tokenized_chord_inv)
			out_chord_timing.append(tokenized_chord_time)
			out_mask.append(tokenized_chord_mask)
		#chords: (B, LEN, 4)
		out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask = torch.tensor(out_chord_root).cuda(), torch.tensor(out_chord_type).cuda(), torch.tensor(out_chord_inv).cuda(), torch.tensor(out_chord_timing).cuda(), torch.tensor(out_mask).cuda()
		embedded_chord = self.chord_embedding_layer(out_chord_root, out_chord_type, out_chord_inv, out_chord_timing)
		return embedded_chord, out_mask
		# return out_chord_root, out_mask


	def forward(self, latents, prompt, beats, chords,chords_time, validation_mode=False):
		device = self.text_encoder.device
		num_train_timesteps = self.noise_scheduler.num_train_timesteps
		self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

		encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
		
		# with torch.no_grad():
		encoded_beats, beat_mask = self.encode_beats(beats) #batch, len_beats, dim; batch, len_beats
		encoded_chords, chord_mask = self.encode_chords(chords,chords_time)


		if self.uncondition:
			mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
			if len(mask_indices) > 0:
				encoder_hidden_states[mask_indices] = 0
				encoded_chords[mask_indices] = 0
				encoded_beats[mask_indices] = 0

		bsz = latents.shape[0]

		if validation_mode:
			timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
		else:
			timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
		
		
		timesteps = timesteps.long()

		noise = torch.randn_like(latents)
		noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

		# Get the target for loss depending on the prediction type
		if self.noise_scheduler.config.prediction_type == "epsilon":
			target = noise
		elif self.noise_scheduler.config.prediction_type == "v_prediction":
			target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
		else:
			raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

		if self.set_from == "random":
			# model_pred = torch.zeros((bsz,8,256,16)).to(device)
			model_pred = self.unet(
				noisy_latents, timesteps, encoder_hidden_states, encoded_beats, encoded_chords,
				encoder_attention_mask=boolean_encoder_mask, beat_attention_mask = beat_mask, chord_attention_mask = chord_mask
			).sample

		elif self.set_from == "pre-trained":
			compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
			model_pred = self.unet(
				compressed_latents, timesteps, encoder_hidden_states, 
				encoder_attention_mask=boolean_encoder_mask
			).sample
			model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

		if self.snr_gamma is None:
			loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
		else:
			# Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
			# Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
			snr = self.compute_snr(timesteps)
			mse_loss_weights = (
				torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
			)
			loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
			loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
			loss = loss.mean()

		return loss

	@torch.no_grad()
	def inference(self, prompt, beats, chords,
			   chords_time, inference_scheduler, 
			   num_steps=20, 
			   guidance_scale=3, num_samples_per_prompt=1, 
				  disable_progress=True):
		device = self.text_encoder.device
		classifier_free_guidance = guidance_scale > 1.0
		batch_size = len(prompt) * num_samples_per_prompt

		if classifier_free_guidance:
			prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
			encoded_beats, beat_mask = self.encode_beats_classifier_free(beats, num_samples_per_prompt) #batch, len_beats, dim; batch, len_beats
			encoded_chords, chord_mask = self.encode_chords_classifier_free(chords, chords_time, num_samples_per_prompt)
		else:
			prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
			prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
			boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

			encoded_beats, beat_mask = self.encode_beats(beats) #batch, len_beats, dim; batch, len_beats
			encoded_beats = encoded_beats.repeat_interleave(num_samples_per_prompt, 0)
			beat_mask = beat_mask.repeat_interleave(num_samples_per_prompt, 0)

			encoded_chords, chord_mask = self.encode_chords(chords,chords_time)
			encoded_chords = encoded_chords.repeat_interleave(num_samples_per_prompt, 0)
			chord_mask = chord_mask.repeat_interleave(num_samples_per_prompt, 0)

		# print(f"encoded_chords:{encoded_chords.shape}, chord_mask:{chord_mask.shape}, prompt_embeds:{prompt_embeds.shape},boolean_prompt_mask:{boolean_prompt_mask.shape} ")
		inference_scheduler.set_timesteps(num_steps, device=device)
		timesteps = inference_scheduler.timesteps

		num_channels_latents = self.unet.in_channels
		latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

		num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
		progress_bar = tqdm(range(num_steps), disable=disable_progress)

		for i, t in enumerate(timesteps):
			# expand the latents if we are doing classifier free guidance
			latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
			latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

			noise_pred = self.unet(
				latent_model_input, t, encoder_hidden_states=prompt_embeds,
				encoder_attention_mask=boolean_prompt_mask, 
				beat_features = encoded_beats, beat_attention_mask = beat_mask, chord_features = encoded_chords,chord_attention_mask = chord_mask
			).sample

			# perform guidance
			if classifier_free_guidance: #should work for beats and chords too
				noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
				noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

			# compute the previous noisy sample x_t -> x_t-1
			latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

			# call the callback, if provided
			if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
				progress_bar.update(1)

		if self.set_from == "pre-trained":
			latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
		return latents

	def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
		shape = (batch_size, num_channels_latents, 256, 16)
		latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
		# scale the initial noise by the standard deviation required by the scheduler
		latents = latents * inference_scheduler.init_noise_sigma
		return latents

	def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
		device = self.text_encoder.device
		batch = self.tokenizer(
			prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
		)
		input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

		with torch.no_grad():
			prompt_embeds = self.text_encoder(
				input_ids=input_ids, attention_mask=attention_mask
			)[0]
				
		prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
		attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

		# get unconditional embeddings for classifier free guidance
		# print(len(prompt), 'this is prompt len')
		uncond_tokens = [""] * len(prompt)

		max_length = prompt_embeds.shape[1]
		uncond_batch = self.tokenizer(
			uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
		)
		uncond_input_ids = uncond_batch.input_ids.to(device)
		uncond_attention_mask = uncond_batch.attention_mask.to(device)

		with torch.no_grad():
			negative_prompt_embeds = self.text_encoder(
				input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
			)[0]
				
		negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
		uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

		# For classifier free guidance, we need to do two forward passes.
		# We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
		prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
		prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
		boolean_prompt_mask = (prompt_mask == 1).to(device)

		return prompt_embeds, boolean_prompt_mask
	

	def encode_beats_classifier_free(self, beats, num_samples_per_prompt):
		with torch.no_grad():
			out_beat = []
			out_beat_timing = []
			out_mask = []
			for beat in beats:
				tokenized_beats,tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
				out_beat.append(tokenized_beats)
				out_beat_timing.append(tokenized_beats_timing)
				out_mask.append(tokenized_beat_mask)
			out_beat, out_beat_timing, out_mask = torch.tensor(out_beat).cuda(), torch.tensor(out_beat_timing).cuda(), torch.tensor(out_mask).cuda() #batch, len_beat
			embedded_beat = self.beat_embedding_layer(out_beat, out_beat_timing)	
			
		embedded_beat = embedded_beat.repeat_interleave(num_samples_per_prompt, 0)
		out_mask = out_mask.repeat_interleave(num_samples_per_prompt, 0)

		uncond_beats = [[[],[]]] * len(beats)

		max_length = embedded_beat.shape[1]
		with torch.no_grad():
			out_beat_unc = []
			out_beat_timing_unc = []
			out_mask_unc = []
			for beat in uncond_beats:
				tokenized_beats, tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
				out_beat_unc.append(tokenized_beats)
				out_beat_timing_unc.append(tokenized_beats_timing)
				out_mask_unc.append(tokenized_beat_mask)
			out_beat_unc, out_beat_timing_unc, out_mask_unc = torch.tensor(out_beat_unc).cuda(), torch.tensor(out_beat_timing_unc).cuda(), torch.tensor(out_mask_unc).cuda() #batch, len_beat
			embedded_beat_unc = self.beat_embedding_layer(out_beat_unc, out_beat_timing_unc)

		embedded_beat_unc = embedded_beat_unc.repeat_interleave(num_samples_per_prompt, 0)
		out_mask_unc = out_mask_unc.repeat_interleave(num_samples_per_prompt, 0)

		embedded_beat = torch.cat([embedded_beat_unc, embedded_beat])
		out_mask = torch.cat([out_mask_unc, out_mask])

		return embedded_beat, out_mask


	def encode_chords_classifier_free(self, chords, chords_time, num_samples_per_prompt):

		with torch.no_grad():
			out_chord_root = []
			out_chord_type = []
			out_chord_inv = []
			out_chord_timing = []
			out_mask = []
			for chord, chord_time in zip(chords,chords_time): #batch loop
				tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
				out_chord_root.append(tokenized_chord_root)
				out_chord_type.append(tokenized_chord_type)
				out_chord_inv.append(tokenized_chord_inv)
				out_chord_timing.append(tokenized_chord_time)
				out_mask.append(tokenized_chord_mask)	
			out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask = torch.tensor(out_chord_root).cuda(), torch.tensor(out_chord_type).cuda(), torch.tensor(out_chord_inv).cuda(), torch.tensor(out_chord_timing).cuda(), torch.tensor(out_mask).cuda()
			embedded_chord = self.chord_embedding_layer(out_chord_root, out_chord_type, out_chord_inv, out_chord_timing)
		
		embedded_chord = embedded_chord.repeat_interleave(num_samples_per_prompt, 0)
		out_mask = out_mask.repeat_interleave(num_samples_per_prompt, 0)

		chords_unc=[[]] * len(chords)
		chords_time_unc=[[]] * len(chords_time)

		max_length = embedded_chord.shape[1]

		with torch.no_grad():
			out_chord_root_unc = []
			out_chord_type_unc = []
			out_chord_inv_unc = []
			out_chord_timing_unc = []
			out_mask_unc = []
			for chord, chord_time in zip(chords_unc,chords_time_unc): #batch loop
				tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
				out_chord_root_unc.append(tokenized_chord_root)
				out_chord_type_unc.append(tokenized_chord_type)
				out_chord_inv_unc.append(tokenized_chord_inv)
				out_chord_timing_unc.append(tokenized_chord_time)
				out_mask_unc.append(tokenized_chord_mask)	
			out_chord_root_unc, out_chord_type_unc, out_chord_inv_unc, out_chord_timing_unc, out_mask_unc = torch.tensor(out_chord_root_unc).cuda(), torch.tensor(out_chord_type_unc).cuda(), torch.tensor(out_chord_inv_unc).cuda(), torch.tensor(out_chord_timing_unc).cuda(), torch.tensor(out_mask_unc).cuda()
			embedded_chord_unc = self.chord_embedding_layer(out_chord_root_unc, out_chord_type_unc, out_chord_inv_unc, out_chord_timing_unc)
		

		embedded_chord_unc = embedded_chord_unc.repeat_interleave(num_samples_per_prompt, 0)
		out_mask_unc = out_mask_unc.repeat_interleave(num_samples_per_prompt, 0)

		embedded_chord = torch.cat([embedded_chord_unc, embedded_chord])
		out_mask = torch.cat([out_mask_unc, out_mask])

		return embedded_chord, out_mask