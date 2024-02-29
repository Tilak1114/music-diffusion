import torch
import torch.nn as nn
import pytorch_lightning as pl
import soundfile as sf
import torch.nn.functional as F
from transformers import get_scheduler
from sample_generation_helper import SampleGeneration
from diffusers.utils.torch_utils import randn_tensor
from unet import UNet
from diffusion import UniformDistribution, VDiffusion


class LatentMusicDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.unet = UNet(config.module_config['unet'])
        
        self.sigmas = UniformDistribution()
        self.v_diffusion = VDiffusion(self.unet, self.sigmas)

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

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=35000
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
                video_embedding):
        true_latent = true_latent.half().to(self.device)
        video_embedding = video_embedding.to(self.device)
        return self.v_diffusion(true_latent, 
                          video_embedding)

    def training_step(self, batch, batch_idx):
        latent_paths, video_embedding_paths = batch

        batch_size = len(latent_paths)

        loaded_latents = []
        for latent_path in latent_paths:
            latent = torch.load(latent_path)
            loaded_latents.append(latent.squeeze(0))

        true_latents = torch.stack(loaded_latents)

        loaded_vid_embs = []
        for vid_emb_path in video_embedding_paths:
            vid_emb = torch.load(vid_emb_path)
            loaded_vid_embs.append(vid_emb.squeeze(0))

        loaded_vid_embs = torch.stack(loaded_vid_embs)

        loss = self.forward(true_latents, loaded_vid_embs)
        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        latent_paths, video_embedding_paths = batch

        batch_size = len(latent_paths)

        loaded_latents = []
        for latent_path in latent_paths:
            latent = torch.load(latent_path)
            loaded_latents.append(latent.squeeze(0))

        true_latents = torch.stack(loaded_latents)

        loaded_vid_embs = []
        for vid_emb_path in video_embedding_paths:
            vid_emb = torch.load(vid_emb_path)
            loaded_vid_embs.append(vid_emb.squeeze(0))

        loaded_vid_embs = torch.stack(loaded_vid_embs)

        val_loss = self.forward(
            true_latents,
            loaded_vid_embs,)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return val_loss

    def val_dataloader(self):
        return self.trainer.datamodule.val_dataloader()

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

    def on_train_epoch_end(self,):
        if self.trainer.global_rank == 0:
            # Get the current epoch
            current_epoch = self.current_epoch
            # Get the loss logged in the last training step
            last_loss = self.trainer.callback_metrics.get("train_loss")
            print(f"Epoch {current_epoch+1}, Train Loss: {last_loss}")

            val_dataloader = self.val_dataloader()

            prompt = next(iter(val_dataloader))[1]
            prompt = "This is an instrumental jam recording of a gear showcase. There is an electric guitar with a clear sound being played with an echo pedal. It gives the recording a dreamy feeling. This track can be used to lift guitar samples with effect for a beat. The chord sequence is D, G. The beat counts to 3. The bpm is 83.0. The key of this song is D minor."

            if self.current_epoch % 5 == 0:
                print(f"Generating for {prompt}")
                wav = self.sample_gen.generate(self.device, prompt)
                out = f"./tmp/output_{self.current_epoch}.wav"
                sf.write(out, wav, samplerate=16000)

        