from data_module import DataModule
import json
from model import LatentMusicDiffusionModel

if __name__ == "__main__":
    file_path = "/data/tilak/projects/music-diffusion/config.json"

    with open(file_path, 'r') as file:
        config_list = json.load(file)

     # Modify the list index for different experiments. Check config.json for details
    config = config_list[1]

    datamodule = DataModule(
        csv_file=config["data_config"]["csv_path"],
        latent_path=config["data_config"]["latent_path"],
        embedding_path=config["data_config"]["video_embedding_path"],
        prompt_embedding_path=config["data_config"]["prompt_embedding_path"],
        batch_size=config["trainer_config"]["batch_size"])

    datamodule.setup()

    device = "cuda:0"

    model = LatentMusicDiffusionModel.load_from_checkpoint(
        "/data/tilak/projects/music-diffusion/checkpoints/epoch=53.ckpt",
        config=config
        ).to(device)
    
    model.eval()

    test_dataloader = datamodule.test_dataloader()

    model.predict(test_dataloader, "/data/tilak/projects/music-diffusion/samples")