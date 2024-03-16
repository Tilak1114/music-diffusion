import os
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, loggers, callbacks
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import LatentMusicDiffusionModel
from torch.utils.data import Dataset, DataLoader
import mlflow
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MusicVideoDataset(Dataset):
	def __init__(self, csv_file, 
			  latent_path, 
			  video_embedding_path):
		self.data_frame = pd.read_csv(csv_file)
		self.latent_path = latent_path
		self.vid_emb_path = video_embedding_path

	def __len__(self):
		return len(self.data_frame[:100])

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

		return f'{audio_file}_{audio_start}_{audio_end}', audio_latent_path, video_emb_path


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
		self.val_size = val_size / (1 - test_size)

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
			batch_size=8
			)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset, 
			batch_size=8
			)


def find_latest_checkpoint(checkpoint_dir):
	checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
	if not checkpoint_files:
		return None
	
	return os.path.join(checkpoint_dir, checkpoint_files[0])

def main():

	file_path = "/data/tilak/projects/music-diffusion/config.json"

	with open(file_path, 'r') as file:
		config_list = json.load(file)
	
	# Modify the list index for different experiments. Check config.json for details
	config = config_list[0]

	with mlflow.start_run(
		run_name=config['name'], 
		description=config['description']):
		
		mlflow.log_param("name", config["name"])
		mlflow.log_param("description", config["description"])
		mlflow.log_param("use_text_conditioning", config["use_text_conditioning"])
		
		# Log nested configurations as JSON strings
		mlflow.log_param("data_config", json.dumps(config["data_config"]))
		mlflow.log_param("pretrained_configs", json.dumps(config["pretrained_configs"]))
		
		# Module configuration
		mlflow.log_param("learning_rate", config["module_config"]["learning_rate"])
		mlflow.log_param("weight_decay", config["module_config"]["weight_decay"])
		mlflow.log_param("gradient_accumulation_steps", config["module_config"]["gradient_accumulation_steps"])
		mlflow.log_param("lr_scheduler", config["module_config"]["lr_scheduler"])
		mlflow.log_param("adam_beta1", config["module_config"]["adam_beta1"])
		mlflow.log_param("adam_beta2", config["module_config"]["adam_beta2"])
		mlflow.log_param("adam_weight_decay", config["module_config"]["adam_weight_decay"])
		mlflow.log_param("adam_eps", config["module_config"]["adam_eps"])
		
		# For deeply nested structures like the UNet config, consider summarizing or logging critical components
		mlflow.log_param("unet_in_channels", config["module_config"]["unet"]["in_channels"])
		mlflow.log_param("unet_block_out_channels", json.dumps(config["module_config"]["unet"]["block_out_channels"]))
		mlflow.log_param("unet_down_block_types", json.dumps(config["module_config"]["unet"]["down_block_types"]))
		mlflow.log_param("unet_attention_head_dim", json.dumps(config["module_config"]["unet"]["attention_head_dim"]))
		mlflow.log_param("unet_up_block_types", json.dumps(config["module_config"]["unet"]["up_block_types"]))
		
		# Trainer configuration
		mlflow.log_param("max_epochs", config["trainer_config"]["max_epochs"])
		mlflow.log_param("devices", config["trainer_config"]["devices"])
		mlflow.log_param("batch_size", config["trainer_config"]["batch_size"])
	
		datamodule = DataModule(csv_file=config["data_config"]["csv_path"], batch_size=config["trainer_config"]["batch_size"])
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
			devices=config["trainer_config"]['devices'],
			logger=tb_logger,
			callbacks=[checkpoint_callback],
			max_epochs=config["trainer_config"]['max_epochs'],
			gradient_clip_val=1.0,
			log_every_n_steps=50,
			enable_progress_bar=True,
			limit_val_batches=100,
			val_check_interval=1.0,
		)

		trainer.fit(model=model, datamodule=datamodule, ckpt_path=last_checkpoint_path)

if __name__ == "__main__":
	mlflow.set_tracking_uri("sqlite:///data/tilak/projects/music-diffusion/mlflow.db")
	main()
