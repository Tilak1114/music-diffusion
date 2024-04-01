import os
from pytorch_lightning import Trainer, loggers, callbacks
from model import SeqLatentMusicDiffusionModel
import json
from sequence_data_module import DataModule
import model_utils as mdl_utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

	file_path = "/data/tilak/projects/music-diffusion/config.json"

	with open(file_path, 'r') as file:
		config_list = json.load(file)
	
	# Modify the list index for different experiments. Check config.json for details
	config = config_list[3]

	datamodule = DataModule(
			csv_file=config["data_config"]["csv_path"], 
			latent_path=config["data_config"]["latent_path"],
			prompt_embedding_path=config["data_config"]["prompt_embedding_path"],
			video_embedding_path=config["data_config"]["video_embedding_path"],
			batch_size=config["trainer_config"]["batch_size"])
	
	datamodule.setup()

	model = SeqLatentMusicDiffusionModel(config)

	tb_logger = loggers.TensorBoardLogger('./logs/')

	checkpoint_callback = callbacks.ModelCheckpoint(
		dirpath='./checkpoints/',
		filename="{epoch:02d}",
		every_n_epochs=1
	)

	last_checkpoint_path = mdl_utils.find_latest_checkpoint(
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
	# mlflow.set_tracking_uri("sqlite:///data/tilak/projects/music-diffusion/mlflow.db")
	main()
