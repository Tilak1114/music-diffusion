from app import MusicDiffusionApp

config_path = "/data/tilak/projects/music-diffusion/config.json"
musicDiffusionApp = MusicDiffusionApp(config_path)

prompt = "Test"
video_path = ''
musicDiffusionApp.inference(prompt, None, None)