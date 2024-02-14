import argparse
import os
import pytorch_lightning as pl
import math
from dataclasses import dataclass

import config

from pytorch_lightning import Trainer, loggers, callbacks

import pandas as pd
from sample_generation_helper import SampleGeneration
import torch

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import soundfile as sf

import tools.torch_tools as torch_tools
from model_utils import build_pretrained_models
from transformers import SchedulerType, get_scheduler
import random

class Text2AudioDataset(Dataset):
    def __init__(self, dataset, text_column, audio_column, beats_column, chords_column, chords_time_column):

        self.inputs = list(dataset[text_column])
        self.audios = list(dataset[audio_column])
        self.beats = list(dataset[beats_column])
        self.chords = list(dataset[chords_column])
        self.chords_time = list(dataset[chords_time_column])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, text, beats, chords in zip(self.indices, self.audios, self.inputs, self.beats, self.chords):
            self.mapper[index] = [audio, text, beats, chords]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4, s5, s6 = self.inputs[index], self.audios[index], self.beats[index], self.chords[index], self.chords_time[index], self.indices[index]
        s2 = '/data/tilak/projects/mustango/data/datashare/'+s2
        return s1, s2, s3, s4, s5, s6

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


class LatentMusicDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        pretrained_config = config.pretrained_configs
        
        self.vae, self.stft = build_pretrained_models(
            pretrained_config["audio_ldm"]
            )
        self.vae.eval()
        self.stft.eval()


        self.model = MusicAudioDiffusion(
            pretrained_config["text_encoder"], 
            pretrained_config["ddpm_scheduler"],
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
        
        if self.args.uncondition_all: # described in Section 5.2 as dropout number 1
            for i in range(len(text)):
                if (random.random()<0.05): #5% chance to drop it all
                    text[i]=""
                    beats[i]=[[],[]]
                    chords[i]=[]
                    chords_time[i]=[]

        if self.args.uncondition_single: #5% chance to drop single ones only... described in Section 5.2 as dropout number 2
            for i in range(len(text)):
                if (random.random()<0.05):
                    text[i]=""
                if (random.random()<0.05):
                    beats[i]=[[],[]]
                if (random.random()<0.05):
                    chords[i]=[]
                    chords_time[i]=[]
        
        with torch.no_grad():
            mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, self.stft)
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

        mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, self.stft)
        mel = mel.unsqueeze(1).to(device)
        true_latent = self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel))
        val_loss = self.forward(true_latent, text, beats, chords, chords_time, validation_mode=True)
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

class Datamodule(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, test_dataset, num_workers=36):
        super().__init__()
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, num_workers=self.num_workers, shuffle=True, batch_size=8, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_dataset, num_workers=self.num_workers, shuffle=False, batch_size=4, collate_fn=self.eval_dataset.collate_fn)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, num_workers=self.num_workers, shuffle=False, batch_size=4, collate_fn=self.test_dataset.collate_fn)

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoint_files:
        return None
    
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def main():
    raw_datasets = load_dataset('json', data_files=config.data_config)
    text_column, audio_column, beats_column, chords_column, chords_time_column = "main_caption", "location", "beats", "chords", "chords_time"

    train_dataset = Text2AudioDataset(
        raw_datasets["train"], 
        text_column, audio_column, 
        beats_column, chords_column, 
        chords_time_column)
    eval_dataset = Text2AudioDataset(
        raw_datasets["validation"], 
        text_column, audio_column, 
        beats_column, chords_column, 
        chords_time_column)
    test_dataset = Text2AudioDataset(
        raw_datasets["test"], 
        text_column, audio_column, 
        beats_column, chords_column, 
        chords_time_column)
    
    datamodule = Datamodule(train_dataset=train_dataset, 
                            eval_dataset=eval_dataset, 
                            test_dataset=test_dataset)

    model = LatentMusicDiffusionModel(args)

    tb_logger = loggers.TensorBoardLogger('./logs/')

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath='./checkpoints/',
        filename="{epoch:02d}",
        every_n_epochs=1
    )

    last_checkpoint_path = find_latest_checkpoint(
        './checkpoints/'
    )

    trainer = Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        devices=args.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_train_epochs,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        limit_val_batches=100,
        val_check_interval=1.0,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=last_checkpoint_path)

if __name__ == "__main__":
    main()
