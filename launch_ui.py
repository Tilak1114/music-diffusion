import gradio as gr
import json
import torch
from model import LatentMusicDiffusionModel

config_path = '/data/tilak/projects/music-diffusion/config.json'

def load_model(model_id):
    with open(config_path, 'r') as file:
        config_list = json.load(file)
    config = config_list[model_id]

    model_wts = ["/data/tilak/projects/music-diffusion/archive/epoch=165.ckpt", "path/to/second/model/weights.ckpt"]
    model_wt = model_wts[model_id]
    model = LatentMusicDiffusionModel(config)
    model.load_state_dict(torch.load(model_wt, map_location=torch.device('cpu')))
    model.eval()
    return model

# Placeholder functions for demonstration
def get_prompt_embedding(prompt):
    pass

def get_video_embedding(video):
    pass

def inference(model_type, text_input, video_input):
    # Conditional logic based on model type
    if model_type == "video_only":
        # Process inputs for Model 1
        pass
    elif model_type == "video_plus_text" or model_type == "Default Model":
        # Process inputs for Model 2 or the default model
        pass
    # Your inference logic here
    # temp
    return "/data/tilak/projects/music-diffusion/experiment_0/tmp/epoch_160/DnrBxSlKd68_0_30.wav"

with gr.Blocks() as app:
    with gr.Row():
        model_type = gr.Dropdown(["video_only", "video_plus_text", "Default Model"], label="Select Model Type")
    with gr.Row():
        text_input = gr.Textbox(lines=3, placeholder="Enter Description Here...")
        video_input = gr.Video(label="Upload Video")
    with gr.Row():
        generate_button = gr.Button("Generate Music")
    output_audio = gr.Audio(label="Generated Audio")

    generate_button.click(inference, inputs=[model_type, text_input, video_input], outputs=[output_audio])

app.launch()
