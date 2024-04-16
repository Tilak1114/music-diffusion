from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning import LightningDataModule

class MusicVideoDataset(Dataset):
	def __init__(self, csv_file, latent_path, video_emb_path, text_emb_path):
		self.data_frame = pd.read_csv(csv_file)
		self.latent_path = latent_path
		self.text_emb_path = text_emb_path
		self.video_emb_path = video_emb_path

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		row = self.data_frame.iloc[idx]
		audio_file = row['Audio File Name'][:-4]
		video_file = row['Video File Name'][:-4]
		is_video = row['isVideo']
		video_start = row['Video Start Time']
		video_end = row['Video End Time']
		# mean_rgb = row['MeanRGBDiff'] if 'MeanRGBDiff' in row and row['MeanRGBDiff'] > 50 else -1
		mean_rgb = row['MeanRGBDiff']

		video_emb_file = f"{video_file}_{video_start}_{video_end}" if is_video else f"{video_file}_0_30"
		video_emb_path = f"{self.video_emb_path}/{video_emb_file}_embedding.pt"
		
		segments = []
		for i in range(1, 4):
			seg = row[f'Segment {i} Reference']
			
			segments.append(f"{self.latent_path}/{seg}_latent.pt")
		
		text_emb_path = f"{self.text_emb_path}/{audio_file}_{row[f'Segment 1 Start']}_{row[f'Segment 1 End']}_prompt.pt"

		save_file_name = f"{audio_file}_{row['Segment 1 Start']}_{row['Segment 3 End']}"

		data_structure = {
			"save_file_name": save_file_name,
			"segments":segments,
			"video_emb_path": video_emb_path,
			"text_emb_path": text_emb_path,
			"mean_rgb": mean_rgb
		}

		return data_structure


class DataModule(LightningDataModule):
	def __init__(self, 
				 csv_file, 
				 latent_path, 
				 prompt_embedding_path,
				 video_embedding_path,
				 batch_size=16, 
				 test_size=0.025, 
				 val_size=0.025):
		super().__init__()
		self.csv_file = csv_file
		self.latent_path = latent_path
		self.prompt_embedding_path = prompt_embedding_path
		self.video_embedding_path = video_embedding_path
		self.batch_size = batch_size
		self.test_size = test_size
		self.val_size = val_size / (1 - test_size)

	def setup(self, stage=None):
		# Load the dataset
		full_dataset = MusicVideoDataset(
			csv_file=self.csv_file,
			latent_path=self.latent_path,
			video_emb_path=self.video_embedding_path,
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
			batch_size=8,
			num_workers=36,
			)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset, 
			batch_size=self.batch_size, 
			num_workers=36
			)
