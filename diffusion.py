from math import pi
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from model_utils import build_pretrained_models
from tqdm import tqdm


class UniformDistribution:
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self,
                 num_samples: int,
                 device: torch.device = torch.device("cpu")
                 ):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(
            num_samples, device=device
        ) + vmin


def extend_dim(x: Tensor, dim: int):
    return x.view(*x.shape + (1,) * (dim - x.ndim))


class VDiffusion(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        sigma_distribution: UniformDistribution,
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, 
                video_embedding: Tensor, 
                prompt_embedding: Tensor = None) -> Tensor:
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(
            num_samples=batch_size,
            device=device
        )

        sigmas_batch = extend_dim(
            sigmas,
            dim=x.ndim
        )

        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy,
                          sigmas,
                          video_embedding,
                          prompt_embedding)

        loss = F.mse_loss(v_pred, v_target)
        return loss


class VSampler:
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        pretrained_model_name = "audioldm-s-full"

        self.vae, self.stft = build_pretrained_models(pretrained_model_name)

        self.vae.eval()
        self.stft.eval()

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def generate_latents(
            self, vid_emb,
            prompt_emb,
            device,
            cfg_scale = 3.0,
            num_steps: int = 100,
            show_progress: bool = False):

        noise_shape = (len(vid_emb), 8, 256, 16)
        x_noisy = torch.randn(noise_shape).to(device)
        vid_emb = vid_emb.to(device)
        prompt_emb = prompt_emb.to(device)

        with torch.no_grad():
            b = x_noisy.shape[0]
            sigmas = torch.linspace(
                1.0, 0.0, num_steps+1, device=x_noisy.device)
            sigmas = repeat(sigmas, "i -> i b", b=b)
            sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
            alphas, betas = self.get_alpha_beta(sigmas_batch)

            for i in range(num_steps):
                v_pred = self.net(x_noisy, sigmas[i], vid_emb, prompt_emb)
                if cfg_scale > 0:
                    v_pred_uncoditional = self.net(x_noisy, sigmas[i], None, None)
                    v_pred = torch.lerp(v_pred_uncoditional, v_pred, cfg_scale)
                    
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return self.latents_to_wave(x_noisy)

    def latents_to_wave(self, latents):
        self.vae = self.vae.to(latents.device)
        mel = self.vae.decode_first_stage(latents)
        wave = self.vae.decode_to_waveform(mel)
        return wave
