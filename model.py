import torch
import pytorch_lightning as pl
import soundfile as sf
import os
from unet import UNet
import mlflow
from clap_metric import ClapMetric
from frechet_audio_distance import FrechetAudioDistance
from diffusion import UniformDistribution, VDiffusion, VSampler
import tempfile
from tqdm import tqdm
import shutil
import json


class LatentMusicDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.unet = UNet(config['module_config']['unet'])

        self.sigmas = UniformDistribution()
        self.v_diffusion = VDiffusion(self.unet,
                                      self.sigmas)
        self.v_sampler = VSampler(self.unet)

        self.frechet = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False,
            use_activation=False,
            verbose=False
        )

        self.clap_metric = ClapMetric()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['module_config']['learning_rate'],
            betas=(
                self.config['module_config']['adam_beta1'],
                self.config['module_config']['adam_beta2']),
            weight_decay=self.config['module_config']['weight_decay'],
            eps=self.config['module_config']['adam_eps']
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
                video_embedding,
                prompt_embedding):
        true_latent = true_latent.half().to(self.device)
        video_embedding = video_embedding.to(
            self.device) if video_embedding != None else None
        prompt_embedding = prompt_embedding.to(
            self.device) if prompt_embedding != None else None

        return self.v_diffusion(true_latent,
                                video_embedding,
                                prompt_embedding)

    def on_train_start(self):
        if self.trainer.global_rank == 0:

            if mlflow.active_run() is not None:
                mlflow.end_run()  # End the currently active run if there is one

            mlflow.set_tracking_uri(
                "file:///data/tilak/projects/music-diffusion/mlflow")
            mlflow.set_experiment(self.config['name'])

            config = self.config

            self.mlflow_run = mlflow.start_run(
                run_name=self.config['name'], description=self.config['description'])

            mlflow.log_param("name", config["name"])
            mlflow.log_param("description", config["description"])

            # Log nested configurations as JSON strings
            mlflow.log_param("data_config", json.dumps(config["data_config"]))
            mlflow.log_param("pretrained_configs", json.dumps(
                config["pretrained_configs"]))

            # Module configuration
            mlflow.log_param(
                "learning_rate", config["module_config"]["learning_rate"])
            mlflow.log_param(
                "weight_decay", config["module_config"]["weight_decay"])
            mlflow.log_param("gradient_accumulation_steps",
                             config["module_config"]["gradient_accumulation_steps"])
            mlflow.log_param(
                "lr_scheduler", config["module_config"]["lr_scheduler"])
            mlflow.log_param(
                "adam_beta1", config["module_config"]["adam_beta1"])
            mlflow.log_param(
                "adam_beta2", config["module_config"]["adam_beta2"])
            mlflow.log_param("adam_weight_decay",
                             config["module_config"]["adam_weight_decay"])
            mlflow.log_param("adam_eps", config["module_config"]["adam_eps"])

            # For deeply nested structures like the UNet config, consider summarizing or logging critical components
            mlflow.log_param("unet_in_channels",
                             config["module_config"]["unet"]["in_channels"])
            mlflow.log_param("unet_block_out_channels", json.dumps(
                config["module_config"]["unet"]["block_out_channels"]))
            mlflow.log_param("unet_down_block_types", json.dumps(
                config["module_config"]["unet"]["down_block_types"]))
            mlflow.log_param("unet_attention_head_dim", json.dumps(
                config["module_config"]["unet"]["attention_head_dim"]))
            mlflow.log_param("unet_up_block_types", json.dumps(
                config["module_config"]["unet"]["up_block_types"]))
            mlflow.log_param(
                "use_text_conditioning", config["module_config"]["unet"]["use_text_conditioning"])

            # Trainer configuration
            mlflow.log_param(
                "max_epochs", config["trainer_config"]["max_epochs"])
            mlflow.log_param("devices", config["trainer_config"]["devices"])
            mlflow.log_param(
                "batch_size", config["trainer_config"]["batch_size"])

    def on_train_end(self):
        if self.global_rank == 0:
            if self.mlflow_run:
                mlflow.end_run()

    def training_step(self, batch, batch_idx):
        audio_file_names, latent_paths, video_embedding_paths, prompt_embedding_paths = batch

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

        loaded_text_embs = []
        for prompt_emb_path in prompt_embedding_paths:
            prompt_emb = torch.load(prompt_emb_path)
            loaded_text_embs.append(prompt_emb)

        loaded_text_embs = torch.stack(loaded_text_embs)

        if torch.rand(1).item() < 0.1:
            loaded_vid_embs = None

        if torch.rand(1).item() < 0.1:
            loaded_text_embs = None

        loss = self.forward(true_latents, loaded_vid_embs, loaded_text_embs)

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        if self.global_rank == 0:
            mlflow.log_metric('train_loss', loss.item(),
                              step=self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_file_names, latent_paths, video_embedding_paths, prompt_embedding_paths = batch

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

        loaded_text_embs = []
        for prompt_emb_path in prompt_embedding_paths:
            prompt_emb = torch.load(prompt_emb_path)
            loaded_text_embs.append(prompt_emb)

        loaded_text_embs = torch.stack(loaded_text_embs)

        val_loss = self.forward(
            true_latents,
            loaded_vid_embs,
            loaded_text_embs
        )

        self.log('val_loss', val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        if self.global_rank == 0 and not self.trainer.sanity_checking:
            mlflow.log_metric('val_loss', val_loss.item(),
                              step=self.current_epoch)

        return val_loss

    def val_dataloader(self):
        return self.trainer.datamodule.val_dataloader()

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

    def on_train_epoch_end(self, outputs=None):

        if self.trainer.global_rank == 0:

            current_epoch = self.current_epoch
            # Get the loss logged in the last training step
            last_loss = self.trainer.callback_metrics.get("train_loss")
            print(f"Epoch {current_epoch+1}, Train Loss: {last_loss}")

            if current_epoch % 5 == 0:
                val_dataloader = self.val_dataloader()
                batch = next(iter(val_dataloader))

                audio_file_names, latent_paths, video_embedding_paths, prompt_embedding_paths = batch

                ground_truth_dir = tempfile.mkdtemp()
                generated_output_dir = tempfile.mkdtemp()

                for i, latent_path in enumerate(latent_paths):
                    latent = torch.load(latent_path).squeeze(0)
                    video_emb = torch.load(video_embedding_paths[i])
                    prompt_emb = torch.load(prompt_embedding_paths[i])

                    ground_truth_wav = self.v_sampler.latents_to_wave(
                        latent.unsqueeze(0))
                    generated_wav = self.v_sampler.generate_latents(
                        video_emb.unsqueeze(0),
                        prompt_emb.unsqueeze(0),
                        self.device
                    )

                    sf.write(os.path.join(ground_truth_dir, f"{audio_file_names[i]}.wav"),
                             ground_truth_wav.squeeze(), samplerate=16000)
                    sf.write(os.path.join(generated_output_dir, f"{audio_file_names[i]}.wav"),
                             generated_wav.squeeze(), samplerate=16000)

                    # Causes a bottleneck in distributed training. Hence only generating one sample
                    break

                fad_score = self.fad(
                    background_dir=ground_truth_dir, eval_dir=generated_output_dir)
                clap_sim_score = self.clap_metric.get_similarity(
                    ground_truth_dir=ground_truth_dir, generated_dir=generated_output_dir)

                # Now log metrics
                mlflow.log_metric('fad', fad_score, step=current_epoch)
                mlflow.log_metric('clap_sim_score', clap_sim_score,
                                  step=self.current_epoch)

                # Log directories to MLflow as artifacts after metrics have been calculated
                mlflow.log_artifacts(
                    ground_truth_dir, artifact_path=f"ground_truth")
                mlflow.log_artifacts(
                    generated_output_dir, artifact_path=f"epoch_{current_epoch}")

                shutil.rmtree(ground_truth_dir)
                shutil.rmtree(generated_output_dir)

                print(f"Fad Score: {fad_score}")
                print(f"Clap Sim Score: {clap_sim_score}")

    def generate_music(self, prompt_emb, video_emb, output_path):
        generated_wav = self.v_sampler.generate_latents(
            video_emb.unsqueeze(0),
            prompt_emb.unsqueeze(0),
            self.device
        )

        output_path = os.path.join(output_path, f"output.wav")

        sf.write(output_path, generated_wav.squeeze(), samplerate=16000)
        
        return output_path

    def inference(self, batch, ground_truth_dir, generated_output_dir, device):
        audio_file_names, latent_paths, video_embedding_paths, prompt_embedding_paths = batch

        if os.path.exists(ground_truth_dir):
            shutil.rmtree(ground_truth_dir)
        os.makedirs(ground_truth_dir)

        if os.path.exists(generated_output_dir):
            shutil.rmtree(generated_output_dir)
        os.makedirs(generated_output_dir)

        for i, latent_path in enumerate(tqdm(latent_paths, desc="Inference")):
            latent = torch.load(latent_path).squeeze(0)
            video_emb = torch.load(video_embedding_paths[i])
            prompt_emb = torch.load(prompt_embedding_paths[i])

            ground_truth_wav = self.v_sampler.latents_to_wave(
                latent.unsqueeze(0))
            generated_wav = self.v_sampler.generate_latents(
                video_emb.unsqueeze(0),
                prompt_emb.unsqueeze(0),
                device
            )

            sf.write(os.path.join(ground_truth_dir, f"{audio_file_names[i]}.wav"),
                     ground_truth_wav.squeeze(), samplerate=16000)
            sf.write(os.path.join(generated_output_dir, f"{audio_file_names[i]}.wav"),
                     generated_wav.squeeze(), samplerate=16000)

        fad_score = self.fad(
            background_dir=ground_truth_dir, eval_dir=generated_output_dir)
        clap_sim_score = self.clap_metric.get_similarity(
            ground_truth_dir=ground_truth_dir, generated_dir=generated_output_dir)

        print(f"Fad Score: {fad_score}")
        print(f"Clap Sim Score: {clap_sim_score}")

    def download_mlflow_artifacts(self, run_id, artifact_path, dest_path):
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(run_id, artifact_path, dest_path)

    def fad(self, background_dir: str, eval_dir: str) -> float:
        fad_score = self.frechet.score(
            background_dir=background_dir,
            eval_dir=eval_dir
        )

        return fad_score
