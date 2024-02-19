import os
import pytorch_lightning as pl
import config
from pytorch_lightning import Trainer, loggers, callbacks
import pandas as pd
from model import LatentMusicDiffusionModel

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        s1, s2, s3, s4, s5, s6, s7, s8 = self.inputs[index], None, self.audios[index], None, self.beats[index], self.chords[index], self.chords_time[index], self.indices[index]
        s3 = '/data/tilak/projects/mustango/data/datashare/'+s3
        base_name = os.path.splitext(os.path.basename(s3))[0]
        s2 = f'/data/tilak/projects/mustango/data/encoded_prompts/{base_name}_prompt_encoding.pt'
        s4 = f'/data/tilak/projects/mustango/data/latents/{base_name}_latent.pt'
        return s1, s2, s3, s4, s5, s6, s7, s8

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


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
        raw_datasets["train_file_path"], 
        text_column, audio_column, 
        beats_column, chords_column, 
        chords_time_column)
    eval_dataset = Text2AudioDataset(
        raw_datasets["val_file_path"], 
        text_column, audio_column, 
        beats_column, chords_column, 
        chords_time_column)
    test_dataset = Text2AudioDataset(
        raw_datasets["test_file_path"], 
        text_column, audio_column, 
        beats_column, chords_column, 
        chords_time_column)
    
    datamodule = Datamodule(train_dataset=train_dataset, 
                            eval_dataset=eval_dataset, 
                            test_dataset=test_dataset)

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
