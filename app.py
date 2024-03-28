import streamlit as st
import mimetypes
from model import LatentMusicDiffusionModel
import json
import torch
import mimetypes
from moviepy.editor import VideoFileClip
from transformers import T5Tokenizer, T5Model
import imageio
import io
import numpy as np
import tempfile
import os
from PIL import Image
import clip

class MusicDiffusionApp:
    def __init__(self, config_path, device):
        self.config_path = config_path
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.text_model = T5Model.from_pretrained("google/flan-t5-large")
        self.vit, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device

    def load_model(self, model_id) -> LatentMusicDiffusionModel:
        with open(self.config_path, 'r') as file:
            config_list = json.load(file)
        config = config_list[model_id]

        checkpoints = ["/data/tilak/projects/music-diffusion/archive/epoch=165.ckpt", 
                       "/data/tilak/projects/music-diffusion/checkpoints/epoch=53.ckpt"]
        ckpt = checkpoints[model_id]
        model = LatentMusicDiffusionModel.load_from_checkpoint(ckpt, config=config)

        model.eval()
        return model

    def get_prompt_embedding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v for k, v in inputs.items()}

        with torch.no_grad():
            encoder_outputs = self.text_model.encoder(**inputs)
            embeddings = encoder_outputs.last_hidden_state.mean(dim=1)

        return embeddings.unsqueeze(0)
    
    def inference(self, prompt, frames):
        prompt_emb = self.get_prompt_embedding(prompt)
        video_emb = self.get_video_embedding(frames)

        print(prompt_emb.shape, video_emb.shape)

        model = self.load_model(1)
        generated_audio_path = model.generate_music(prompt_emb, video_emb, "./outputs")
        return generated_audio_path

    def get_video_embedding(self, frames):
        embeddings = []
        for frame in frames:
            image = self.preprocess(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings.append(self.vit.encode_image(image))
        
        return torch.stack(embeddings).mean(dim=0).unsqueeze(0)

def app(musicDiffusionApp: MusicDiffusionApp):
    st.title("Harmonizing Pixels and Music")

    # Collect text input
    text_input = st.text_area("Enter Description of the music:", height=150)

    # File uploader for image or video
    file_input = st.file_uploader("Upload a Video or an Image", type=["mp4", "jpg", "png"])

    video_duration_valid = False

    frames = []

    # Always active "Generate Music" button
    if st.button("Generate Music"):
        # Check if either text input or file is provided
        if text_input.strip() == "" and file_input is None:
            st.error("Please provide a text description or upload a file.")
        else:
            mime_type = None if not file_input else mimetypes.guess_type(file_input.name)[0]

            if file_input is not None:
                if mime_type.startswith("image/"):
                    st.image(file_input)

                elif mime_type.startswith("video/"):
                    st.video(file_input)
            
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                        tmpfile.write(file_input.read())
                        temp_path = tmpfile.name
    
                    try:
                        reader = imageio.get_reader(temp_path, 'ffmpeg')
                        video_duration = reader.get_meta_data()['duration']
                        
                        if video_duration > 30:
                            video_duration_valid = False
                            st.error("Video is longer than 30 seconds.")
                            return
                        else:
                            video_duration_valid = True
                            fps = reader.get_meta_data()['fps']  # Frames per second
                            seconds = 0  # Initialize a counter for seconds

                            # Extract one frame per second
                            for i, frame in enumerate(reader):
                                if i % int(fps) == 0:  # Check if the frame corresponds to a new second
                                    frame_image = Image.fromarray(frame)
                                    frames.append(frame_image)  # Append the frame as a PIL Image
                                    seconds += 1
                                    if seconds >= 30:  # Stop after 30 seconds
                                        break
                        
                    finally:
                        os.unlink(temp_path)
                                    
            # Use st.spinner to show a loading indicator while processing
            if (file_input != None and video_duration_valid) or (file_input == None):
                with st.spinner('Generating music, please wait...'):
                    generated_path = musicDiffusionApp.inference(text_input, frames)

                if generated_path:
                    st.audio(generated_path, format='audio/wav', start_time=0)
                    st.success("Done!")
                else:
                    st.error("Failed to generate music. Please check your inputs.")

if __name__ == "__main__":
    config_path = "/data/tilak/projects/music-diffusion/config.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    musicDiffusionApp = MusicDiffusionApp(config_path, device)
    app(musicDiffusionApp)
    