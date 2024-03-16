import librosa
import torch
import os
from transformers import ClapAudioModelWithProjection, ClapProcessor

class ClapMetric:
    def __init__(self):
        self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    
    def get_clap_embedding(self, audio_file):
        sampling_rate = 48000
        audio, _ = librosa.load(audio_file, sr=sampling_rate)
        inputs = self.processor(
            audios=audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.audio_embeds
    
    def get_similarity(self, ground_truth_dir, generated_dir):
        cosine_similarities = []
        
        # Iterate over ground truth audio files
        for gt_filename in os.listdir(ground_truth_dir):
            gt_filepath = os.path.join(ground_truth_dir, gt_filename)
            
            # Find the corresponding generated file
            gen_filepath = os.path.join(generated_dir, gt_filename)
            
            # Compute CLAP embeddings for both files
            gt_embedding = self.get_clap_embedding(gt_filepath)
            gen_embedding = self.get_clap_embedding(gen_filepath)
            
            # Compute cosine similarity and append to list
            cosine_similarity = torch.nn.functional.cosine_similarity(
                gt_embedding, 
                gen_embedding, 
                dim=1
            )
            cosine_similarities.append(cosine_similarity.item())
        
        average_similarity = torch.mean(torch.Tensor(cosine_similarities))
        return average_similarity.item()
