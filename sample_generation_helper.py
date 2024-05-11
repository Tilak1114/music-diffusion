import torch
import numpy as np
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel
from modelling_deberta_v2 import DebertaV2ForTokenClassificationRegression

from diffusers import DDPMScheduler
from model_utils import build_pretrained_models


class MusicFeaturePredictor:
    def __init__(self, path, cache_dir=None, local_files_only=False):
        self.beats_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.beats_model = DebertaV2ForTokenClassificationRegression.from_pretrained(
            "microsoft/deberta-v3-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.beats_model.eval()

        beats_ckpt = f"{path}/beats/microsoft-deberta-v3-large.pt"
        beats_weight = torch.load(beats_ckpt, map_location="cpu")
        self.beats_model.load_state_dict(beats_weight)

        self.chords_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.chords_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.chords_model.eval()
        self.chords_model

        chords_ckpt = f"{path}/chords/flan-t5-large.bin"
        chords_weight = torch.load(chords_ckpt, map_location="cpu")
        self.chords_model.load_state_dict(chords_weight)

    def generate_beats(self, prompt):
        tokenized = self.beats_tokenizer(
            prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized = {k: v.to(self.beats_model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            out = self.beats_model(**tokenized)

        max_beat = (
            1 + torch.argmax(out["logits"][:, 0, :], -1).detach().cpu().numpy()
        ).tolist()[0]
        intervals = (
            out["values"][:, :, 0]
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
            .round(4)
            .tolist()
        )

        intervals = np.cumsum(intervals)
        predicted_beats_times = []
        for t in intervals:
            if t < 10:
                predicted_beats_times.append(round(t, 2))
            else:
                break
        predicted_beats_times = list(np.array(predicted_beats_times)[:50])

        if len(predicted_beats_times) == 0:
            predicted_beats = [[], []]
        else:
            beat_counts = []
            for i in range(len(predicted_beats_times)):
                beat_counts.append(float(1.0 + np.mod(i, max_beat)))
            predicted_beats = [[predicted_beats_times, beat_counts]]

        return max_beat, predicted_beats_times, predicted_beats

    def generate(self, prompt):
        max_beat, predicted_beats_times, predicted_beats = self.generate_beats(prompt)

        chords_prompt = "Caption: {} \\n Timestamps: {} \\n Max Beat: {}".format(
            prompt,
            " , ".join([str(round(t, 2)) for t in predicted_beats_times]),
            max_beat,
        )

        tokenized = self.chords_tokenizer(
            chords_prompt,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.chords_model.device) for k, v in tokenized.items()}

        generated_chords = self.chords_model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            min_length=8,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=1,
        )

        generated_chords = self.chords_tokenizer.decode(
            generated_chords[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).split(" n ")

        predicted_chords, predicted_chords_times = [], []
        for item in generated_chords:
            c, ct = item.split(" at ")
            predicted_chords.append(c)
            predicted_chords_times.append(float(ct))

        return predicted_beats, predicted_chords, predicted_chords_times


class SampleGeneration:
    def __init__(
        self,
        model,
        name="declare-lab/mustango",
        cache_dir=None,
        local_files_only=False,
    ):
        path = snapshot_download(repo_id=name, cache_dir=cache_dir)

        self.music_model = MusicFeaturePredictor(
            path, cache_dir=cache_dir, local_files_only=local_files_only
        )

        pretrained_model_name = "audioldm-s-full"

        self.vae, self.stft = build_pretrained_models(pretrained_model_name) 
        self.model = model

        pretrained_model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name)

        self.vae.eval()
        self.stft.eval()
        self.model.eval()

        self.scheduler = DDPMScheduler.from_pretrained(
            'stabilityai/stable-diffusion-2-1', subfolder="scheduler"
        )
    
    def encode_text(self, prompt, device):
        batch = self.tokenizer(
            prompt, 
            max_length=171, 
            padding='max_length', 
            truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(
            device), batch.attention_mask.to(device)  # cuda

        with torch.no_grad():
            self.text_encoder = self.text_encoder.to(device)
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(
            device)  # batch, len_text
        return encoder_hidden_states, boolean_encoder_mask

    def generate(self, device, prompt, steps=200, samples=1):
        """Genrate music for a single prompt string."""

        encoded_prompt, _ = self.encode_text(prompt, device)
        self.vae = self.vae.to(device)
       
        with torch.no_grad():
            beats, chords, chords_times = self.music_model.generate(prompt)
            latents = self.model.inference(
                encoded_prompt,
                beats,
                [chords],
                [chords_times],
                self.scheduler,
                device,
                steps,
                samples,
            )
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)

        return wave[0]
