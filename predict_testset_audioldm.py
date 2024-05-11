from diffusers import AudioLDM2Pipeline
import torch
import scipy
import os
from sequence_data_module import DataModule
import json
import random
from tqdm import tqdm

repo_id = "cvssp/audioldm2-music"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

SAVE_DIR = '/data/tilak/projects/music-diffusion/samples/audioldm_generated'
PROMPT_FILE = '/data/tilak/scripts/predictions.json'

def generate_audio(prompt, save_file_name, directory):
	audio = pipe(prompt, num_inference_steps=100, audio_length_in_s=30.0).audios[0]
	full_path = os.path.join(directory, save_file_name)
	scipy.io.wavfile.write(f"{full_path}.wav", rate=16000, data=audio)

def get_matching_prompt(data, save_file_name):
    parts = save_file_name.split('_')
    
    start, end = int(parts[-2]), int(parts[-1])
    base_name = '_'.join(parts[:-2])
    
    # Generate all possible key segments within the given range
    possible_keys = [f"{base_name}_{i}_{i+10}" for i in range(int(start), int(end), 10)]
    
    # Find all matching keys
    matching_keys = [key for key in possible_keys if key in data]
    
    # Randomly select one prompt if there are multiple matches
    if matching_keys:
        selected_key = random.choice(matching_keys)
        return data[selected_key]
    else:
        return ""

if __name__ == "__main__":
	file_path = "/data/tilak/projects/music-diffusion/config.json"
	
	with open(file_path, 'r') as file:
		config_list = json.load(file)
	
	with open(PROMPT_FILE, 'r') as file:
		prompt_file = json.load(file)

	# Modify the list index for different experiments. Check config.json for details
	config = config_list[3]

	datamodule = DataModule(
			csv_file=config["data_config"]["csv_path"], 
			latent_path=config["data_config"]["latent_path"],
			prompt_embedding_path=config["data_config"]["prompt_embedding_path"],
			video_embedding_path=config["data_config"]["video_embedding_path"],
			batch_size=config["trainer_config"]["batch_size"])
	
	datamodule.setup()

	test_dataloader = datamodule.test_dataloader()

	for batch_idx, batch in tqdm(enumerate(test_dataloader), desc="Processing Batches"):
		save_file_names = batch['save_file_name']
		prompt_emb_paths = batch['text_emb_path']

		for save_file in save_file_names:
			prompt = get_matching_prompt(prompt_file, save_file)
			generate_audio(prompt, save_file, SAVE_DIR)
