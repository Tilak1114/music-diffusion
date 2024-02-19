import os
import pytorch_lightning as pl
import config
from pytorch_lightning import Trainer, loggers, callbacks
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, T5EncoderModel

class TextEncodeModule(pl.LightningModule):
    def __init__(self, save_path):
        super().__init__()

        pretrained_model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
       
        self.save_path = save_path

    def forward(self, x):
        return self.encode_text(x)
        
    
    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, 
            max_length=171, 
            padding='max_length', 
            truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(
            device), batch.attention_mask.to(device)  # cuda

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(
            device)  # batch, len_text
        return encoder_hidden_states, boolean_encoder_mask

    def predict_step(self, batch, batch_idx):
        prompts, audio_paths = batch
        
        for idx, prompt in enumerate(prompts):
            # Extract base name without extension and directory
            base_name = os.path.splitext(os.path.basename(audio_paths[idx]))[0]
            save_path = f"{self.save_path}/{base_name}_prompt_encoding.pt"
            
            encoder_hidden_states, boolean_encoder_mask = self.forward(prompt)
            # Ensure the save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save({
                "encoder_hidden_states": encoder_hidden_states.cpu(),
                "boolean_encoder_mask": boolean_encoder_mask.cpu()
            }, save_path)
    
    def configure_optimizers(self):
        return None

class PromptDataset(Dataset):
    def __init__(self, dataset, text_column, audio_column):
        self.inputs = list(dataset[text_column])
        self.audios = list(dataset[audio_column])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.audios[idx]

def main():

    raw_datasets = load_dataset('json', data_files=config.data_config)
    text_column, audio_column, beats_column, chords_column, chords_time_column = "main_caption", "location", "beats", "chords", "chords_time"

    train_dataset = PromptDataset(
        raw_datasets["train_file_path"], 
        text_column, audio_column)
    
    eval_dataset = PromptDataset(
        raw_datasets["val_file_path"], 
        text_column, audio_column)
    
    test_dataset = PromptDataset(
        raw_datasets["test_file_path"], 
        text_column, audio_column)

    datasets = [train_dataset, eval_dataset, test_dataset]

    model = TextEncodeModule('/data/tilak/projects/mustango/data/encoded_prompts')


    for dataset in datasets:
        dataloader = DataLoader(dataset, 
                                batch_size=128, 
                                num_workers=36, 
                                shuffle=False)

        trainer = Trainer(
            accelerator='gpu',
            strategy='ddp_find_unused_parameters_true',
            devices=-1,
            enable_progress_bar=True,
        )

        trainer.predict(model=model, dataloaders=dataloader)

if __name__ == "__main__":
    main()
