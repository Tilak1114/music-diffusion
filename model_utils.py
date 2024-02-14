from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata, download_checkpoint
import os
import torch


def build_pretrained_models(name):
	
	ckpt_path = get_metadata()[name]["path"]
	if not os.path.exists(ckpt_path):
		download_checkpoint(name)

	checkpoint = torch.load(ckpt_path, map_location="cpu")
	scale_factor = checkpoint["state_dict"]["scale_factor"].item()

	vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

	config = default_audioldm_config(name)
	vae_config = config["model"]["params"]["first_stage_config"]["params"]
	vae_config["scale_factor"] = scale_factor

	vae = AutoencoderKL(**vae_config)
	vae.load_state_dict(vae_state_dict)

	fn_STFT = TacotronSTFT(
		config["preprocessing"]["stft"]["filter_length"],
		config["preprocessing"]["stft"]["hop_length"],
		config["preprocessing"]["stft"]["win_length"],
		config["preprocessing"]["mel"]["n_mel_channels"],
		config["preprocessing"]["audio"]["sampling_rate"],
		config["preprocessing"]["mel"]["mel_fmin"],
		config["preprocessing"]["mel"]["mel_fmax"],
	)

	vae.eval()
	fn_STFT.eval()
	return vae, fn_STFT