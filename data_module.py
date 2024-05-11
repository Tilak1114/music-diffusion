from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning import LightningDataModule

class MusicVideoDataset(Dataset):
	def __init__(self, csv_file, 
			  latent_path, 
			  video_embedding_path,
			  text_emb_path):
		self.data_frame = pd.read_csv(csv_file)
		self.latent_path = latent_path
		self.vid_emb_path = video_embedding_path
		self.text_emb_path = text_emb_path

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
		
		text_emb_path = f"{self.text_emb_path}/{audio_file}_{audio_start}_{audio_end}_prompt.pt"

		return f'{audio_file}_{audio_start}_{audio_end}', audio_latent_path, video_emb_path, text_emb_path


class DataModule(LightningDataModule):
	def __init__(self, 
				 csv_file, 
				 latent_path, 
				 embedding_path,
				 prompt_embedding_path,
				 batch_size=16, 
				 test_size=0.025, 
				 val_size=0.025):
		super().__init__()
		self.csv_file = csv_file
		self.latent_path = latent_path
		self.embedding_path = embedding_path
		self.prompt_embedding_path = prompt_embedding_path
		self.batch_size = batch_size
		self.test_size = test_size
		self.val_size = val_size / (1 - test_size)

	def setup(self, stage=None):
		# Load the dataset
		full_dataset = MusicVideoDataset(
			csv_file=self.csv_file,
			latent_path=self.latent_path,
			video_embedding_path=self.embedding_path,
			text_emb_path=self.prompt_embedding_path
			)
		
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
			shuffle=True,
			num_workers=36
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
