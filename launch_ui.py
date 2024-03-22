import streamlit as st
import mimetypes
from model import LatentMusicDiffusionModel
import json
import torch
import mimetypes
import tempfile
import os
from transformers import T5Tokenizer, T5Model

class MusicDiffusionApp:
    def __init__(self, config_path):
        self.config_path = config_path
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.text_model = T5Model.from_pretrained("google/flan-t5-large")

    def load_model(self, model_id) -> LatentMusicDiffusionModel:
        with open(self.config_path, 'r') as file:
            config_list = json.load(file)
        config = config_list[model_id]

        checkpoints = ["/data/tilak/projects/music-diffusion/archive/epoch=165.ckpt", 
                       "/data/tilak/projects/music-diffusion/checkpoints/epoch=41.ckpt"]
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

        return embeddings
    
    def inference(self, model_id, prompt, file_data, mime_type):
        prompt_emb = self.get_prompt_embedding(prompt)
        video_emb = None

        mime_type, _ = mimetypes.guess_type(file_data.name)

        is_video = mime_type.startswith('video/')
        
        # Convert binary data to a temporary file if it's a video
        if is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file_data.read())
                tmp_path = tmp.name
                video_emb = self.get_video_embedding(tmp_path)
        else:
            video_emb = None

        model = self.load_model(model_id)
        generated_audio_path = model.generate_music(prompt_emb, video_emb, "./outputs")
        return generated_audio_path

    def get_video_embedding(self, video):
        # Placeholder for video embedding logic
        return None

def app(musicDiffusionApp: MusicDiffusionApp):
    st.title("Harmonizing Pixels and Music")

    model_id = st.selectbox("Select Model Type", [0, 1], format_func=lambda x: "Model "+str(x+1))

    text_input = st.text_area("Enter Description of the music:", height=150)

    # File uploader for image or video
    file_input = st.file_uploader("Upload a Video or an Image", type=["mp4", "jpg", "png"])
    # Check if a file is uploaded
    if file_input is not None:
        mime_type = mimetypes.guess_type(file_input.name)[0]

        # Display the uploaded image or video
        if mime_type.startswith("image/"):
            st.image(file_input)
        elif mime_type.startswith("video/"):
            st.video(file_input)

        if st.button("Generate Music"):
            st.write("Processing...")
            generated_path = musicDiffusionApp.inference(model_id, text_input, file_input, mime_type)

            if generated_path:
                st.audio(generated_path, format='audio/wav', start_time=0)
                st.write("Done!")
            else:
                st.write("Failed to generate music. Please check your inputs.")

if __name__ == "__main__":
    config_path = "/data/tilak/projects/music-diffusion/config.json"
    musicDiffusionApp = MusicDiffusionApp(config_path)
    app(musicDiffusionApp)