from math import pi
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
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
        sigma_distribution: UniformDistribution
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, video_embedding: Tensor) -> Tensor:
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
                          video_embedding)

        loss = F.mse_loss(v_pred, v_target)
        return loss


class LinearSchedule(nn.Module):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)


class VSampler(nn.Module):

    def __init__(self, net: nn.Module,
                 schedule=LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_noisy: Tensor,
        num_steps: int,
        show_progress: bool = False
    ) -> Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps),
                            disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i])
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(
                f"Sampling (noise={sigmas[i+1, 0]:.2f})")

        return x_noisy
