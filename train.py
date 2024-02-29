import os
import pytorch_lightning as pl
import config
import torch
from pytorch_lightning import Trainer, loggers, callbacks
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import LatentMusicDiffusionModel
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MusicVideoDataset(Dataset):
	def __init__(self, csv_file,):
		self.data_frame = pd.read_csv(csv_file)
		self.latent_path = "/data/tilak/scripts/master_data/data/latents"
		self.vid_emb_path = "/data/tilak/scripts/master_data/data/video_embeddings"

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		row = self.data_frame.iloc[idx]
		audio_file = row['Audio File Name'][:-4]
		audio_start = row['Audio Start Time']
		audio_end = row['Audio End Time']
		video_file = row['Video File Name'][:-4]
		video_start = row['Video Start Time']
		video_end = row['Video End Time']
		is_video = row['isVideo']

		audio_latent_path = f"{self.latent_path}/{audio_file}_{audio_start}_{audio_end}_latent.pt"
		
		if is_video:
			video_emb_path = f"{self.vid_emb_path}/{video_file}_{video_start}_{video_end}_embedding.pt"
		else:
			video_emb_path = f"{self.vid_emb_path}/{video_file}_0_30_embedding.pt"

		return audio_latent_path, video_emb_path


class DataModule(pl.LightningDataModule):
	def __init__(self, 
				 csv_file, 
				 batch_size=16, 
				 test_size=0.025, 
				 val_size=0.025):
		super().__init__()
		self.csv_file = csv_file
		self.batch_size = batch_size
		self.test_size = test_size
		self.val_size = val_size / (1 - test_size)  # Adjust val_size based on test_size to maintain proportion after test split

	def setup(self, stage=None):
		# Load the dataset
		full_dataset = MusicVideoDataset(
			csv_file=self.csv_file,)
		
		# Split data into train+val and test sets
		train_val_indices, test_indices = train_test_split(range(len(full_dataset)), 
														   test_size=self.test_size,
															random_state=42)
		train_val_dataset = torch.utils.data.Subset(full_dataset, 
													train_val_indices)
		self.test_dataset = torch.utils.data.Subset(full_dataset, 
													test_indices)
		
		# Split train+val into train and val sets
		train_indices, val_indices = train_test_split(range(len(train_val_dataset)), test_size=self.val_size, random_state=42)
		self.train_dataset = torch.utils.data.Subset(train_val_dataset, train_indices)
		self.val_dataset = torch.utils.data.Subset(train_val_dataset, val_indices)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset, 
			batch_size=self.batch_size, 
			shuffle=True
			)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset, 
			batch_size=self.batch_size
			)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset, 
			batch_size=self.batch_size
			)


def find_latest_checkpoint(checkpoint_dir):
	checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
	if not checkpoint_files:
		return None
	
	return os.path.join(checkpoint_dir, checkpoint_files[0])

def main():
	
	datamodule = DataModule(csv_file="/data/tilak/scripts/master_data/final2.csv")
	datamodule.setup()

	model = LatentMusicDiffusionModel(config)

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
		precision=16,
		devices=config.trainer_config['devices'],
		logger=tb_logger,
		callbacks=[checkpoint_callback],
		max_epochs=config.trainer_config['max_epochs'],
		gradient_clip_val=1.0,
		log_every_n_steps=50,
		enable_progress_bar=True,
		limit_val_batches=100,
		val_check_interval=1.0,
	)

	trainer.fit(model=model, datamodule=datamodule, ckpt_path=last_checkpoint_path)

if __name__ == "__main__":
	main()
