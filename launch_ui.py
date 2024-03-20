import gradio as gr
import json
import torch
from model import LatentMusicDiffusionModel
from transformers import T5Tokenizer, T5Model

config_path = '/data/tilak/projects/music-diffusion/config.json'

def load_model(model_id) -> LatentMusicDiffusionModel:
    with open(config_path, 'r') as file:
        config_list = json.load(file)
    config = config_list[model_id]

    checkpoints = ["/data/tilak/projects/music-diffusion/archive/epoch=165.ckpt", "path/to/second/model/weights.ckpt"]
    ckpt = checkpoints[model_id]
    model = LatentMusicDiffusionModel.load_from_checkpoint(ckpt, config)
    
    model.eval()
    return model

# Placeholder functions for demonstration
def get_prompt_embedding(prompt):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5Model.from_pretrained("google/flan-t5-large")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v for k, v in inputs.items()}

    # For T5, just using the encoder should suffice for getting text embeddings
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
        # Taking the mean of the last hidden state to get a single vector representation
        embeddings = encoder_outputs.last_hidden_state.mean(dim=1)

    return embeddings

def get_video_embedding(video):
    pass

def inference(model_type, text_input, video_input):
    vid_emb = get_video_embedding(video_input)
    prompt_emb = get_prompt_embedding(text_input)
    # Conditional logic based on model type
    if model_type == "video_only":
        # Process inputs for Model 1
        model = load_model(0)
        output_path = model.generate_music(
            prompt_emb=prompt_emb, 
            video_emb=vid_emb, 
            output_path="/data/tilak/projects/music-diffusion/outputs"
            )
    elif model_type == "video_plus_text" or model_type == "Default Model":
        # model = load_model(1)
        pass
    # Your inference logic here
    # temp
    return output_path

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
