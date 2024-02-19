import os
import pytorch_lightning as pl
import config
from pytorch_lightning import Trainer, loggers, callbacks
import pandas as pd
from tools import torch_tools
from torch.utils.data import Dataset, DataLoader
import torch
from model_utils import build_pretrained_models


class LatentPrecomputeModule(pl.LightningModule):
    def __init__(self, save_path):
        super().__init__()

        pretrained_model_name = "audioldm-s-full"

        self.vae, self.stft = build_pretrained_models(pretrained_model_name)
        self.vae.eval()
        self.stft.eval()
       
        self.save_path = save_path

    def forward(self, x):
        # Assuming 'x' is your input data
        mel, _ = self.stft(x)  # Compute STFT or any required preprocessing
        latent = self.vae.encode(mel)  # Get the latent representation
        return latent

    def predict_step(self, batch, batch_idx):
        audios = batch
        target_length = int(10 * 102.4)
        
        for idx, audio_path in enumerate(audios):
            with torch.no_grad():
                # Assuming `torch_tools.wav_to_fbank` can process individual files; adjust as needed
                mel, _, waveform = torch_tools.wav_to_fbank(
                    [audio_path],  # Now processing one file at a time
                    target_length,
                    self.stft
                )
                
                mel = mel.unsqueeze(0).to(self.device)
                true_latent = self.vae.get_first_stage_encoding(
                    self.vae.encode_first_stage(mel))

            # Extract base name without extension and directory
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            save_path = f"{self.save_path}/{base_name}_latent.pt"

            # Ensure the save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the latent vector for the current file
            torch.save(true_latent.cpu(), save_path)

    def configure_optimizers(self):
        # This method must be defined, but you can return None if not training
        return None

class AudioFilePathDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Return the file path of the audio file
        return self.audio_files[idx]

dataset = AudioFilePathDataset(directory='/data/tilak/projects/mustango/data/datashare/data_aug2')
dataloader = DataLoader(dataset, batch_size=128, num_workers=36, shuffle=False)

def main():
   
    model = LatentPrecomputeModule('/data/tilak/projects/mustango/data/latents')

    trainer = Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        devices=-1,
        enable_progress_bar=True,
    )

    trainer.predict(model=model, dataloaders=dataloader)

if __name__ == "__main__":
    main()
