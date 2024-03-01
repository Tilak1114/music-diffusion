import torch
import torch.nn as nn
import pytorch_lightning as pl
import soundfile as sf
import torch.nn.functional as F
from transformers import get_scheduler
import os
from sample_generation_helper import SampleGeneration
from diffusers.utils.torch_utils import randn_tensor
from unet import UNet
from frechet_audio_distance import FrechetAudioDistance
from diffusion import UniformDistribution, VDiffusion, VSampler


class LatentMusicDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.unet = UNet(config.module_config['unet'])
        
        self.sigmas = UniformDistribution()
        self.v_diffusion = VDiffusion(self.unet, self.sigmas)
        self.v_sampler = VSampler(self.unet)

        self.frechet = FrechetAudioDistance(
            model_name="pann",
            use_pca=False,
            use_activation=False,
            verbose=False
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
        # lr_scheduler = get_scheduler(
        #     name='linear',
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=35000
        # )

        return optimizer

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
            loaded_vid_embs.append(vid_emb)

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
            loaded_vid_embs.append(vid_emb)

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

            if self.current_epoch % 5 == 0:

                val_dataloader = self.val_dataloader()
                sample_batch = next(iter(val_dataloader))

                latent_paths, video_embedding_paths = sample_batch

                loaded_latents = []
                for latent_path in latent_paths:
                    latent = torch.load(latent_path)
                    loaded_latents.append(latent.squeeze(0))

                latents = torch.stack(loaded_latents)

                loaded_vid_embs = []
                for vid_emb_path in video_embedding_paths:
                    vid_emb = torch.load(vid_emb_path)
                    loaded_vid_embs.append(vid_emb)

                video_embeddings = torch.stack(loaded_vid_embs)

                ground_truth_wav = self.v_sampler.latents_to_wave(latents)

                for i, latent_path in enumerate(latent_paths):
                    filename = os.path.splitext(os.path.basename(latent_path))[0].rsplit('_latent', 1)[0]
                    out_dir = "./ground_truth"
                    os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist
                    out = f"{out_dir}/{filename}.wav"
                    sf.write(out, ground_truth_wav[i], samplerate=16000)

                wav = self.v_sampler.generate_latents(video_embeddings, self.device)

                for i, video_emb_path in enumerate(video_embedding_paths):
                    filename = os.path.splitext(os.path.basename(video_emb_path))[0].rsplit('_embedding', 1)[0]
                    out_dir = f"./tmp/epoch_{self.current_epoch}"
                    os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist
                    out = f"{out_dir}/{filename}.wav"
                    sf.write(out, wav[i], samplerate=16000)

                fad_score = self.fad(
                    background_dir='./ground_truth/',
                    eval_dir=f"./tmp/epoch_{self.current_epoch}/"
                )

                self.log('fad', fad_score)
                print(f"Fad Score: {fad_score}")

    def fad(self, background_dir: str, eval_dir: str) -> float:
        fad_score = self.frechet.score(
            background_dir=background_dir,
            eval_dir=eval_dir
        )

        return fad_score
        