#!/usr/bin/env python3
"""
AE5510: Expectation Over Transformation (EOT) Physical Sampler
==============================================================
Module: adv_camouflage/attack/eot.py

Applies differentiable physical domain randomization:
  1. In-plane rotation jitter (-20 deg to +20 deg)
  2. Perspective shear jitter (-0.05 to +0.05)
  3. Scale / Zoom factor (0.90 to 1.10)
  4. Illumination variation (0.85 to 1.15)
  5. Atmospheric haze model: I = (1 - h) * I + h * A_air
  6. Camera sensor gamma jitter (1.8 to 2.4)
  7. Sensor Gaussian noise (sigma 0 to 0.02)
  8. Defocus / Atmospheric blur (sigma 0 to 0.8 px)
  9. Color quantization STE approximation
"""

from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_camouflage.attack.render import gaussian_blur_2d


class EOTSampler(nn.Module):
    def __init__(
        self,
        max_rotation_deg: float = 20.0,
        max_shear: float = 0.05,
        min_scale_jitter: float = 0.90,
        max_scale_jitter: float = 1.10,
        illumination_min: float = 0.85,
        illumination_max: float = 1.15,
        haze_max: float = 0.12,
        gamma_min: float = 1.8,
        gamma_max: float = 2.4,
        noise_sigma_max: float = 0.02,
        blur_sigma_max: float = 0.80,
    ):
        super().__init__()
        self.max_rot_rad = math.radians(max_rotation_deg)
        self.max_shear = max_shear
        self.min_scale = min_scale_jitter
        self.max_scale = max_scale_jitter
        self.illum_min = illumination_min
        self.illum_max = illumination_max
        self.haze_max = haze_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.noise_max = noise_sigma_max
        self.blur_max = blur_sigma_max

    def sample_placement_jitter(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Samples perturbed placement parameters:
        theta = [tx, ty, log_s, phi, hx, hy]
        """
        device = theta.device
        dtype = theta.dtype

        # Rotation jitter
        d_phi = (torch.rand(1, device=device, dtype=dtype) * 2.0 - 1.0) * self.max_rot_rad
        # Perspective shear jitter
        d_hx = (torch.rand(1, device=device, dtype=dtype) * 2.0 - 1.0) * self.max_shear
        d_hy = (torch.rand(1, device=device, dtype=dtype) * 2.0 - 1.0) * self.max_shear
        # Scale jitter
        scale_factor = torch.empty(1, device=device, dtype=dtype).uniform_(self.min_scale, self.max_scale)
        d_log_s = torch.log(scale_factor)

        perturbed_theta = theta.clone()
        perturbed_theta[2] = perturbed_theta[2] + d_log_s
        perturbed_theta[3] = perturbed_theta[3] + d_phi
        perturbed_theta[4] = perturbed_theta[4] + d_hx
        perturbed_theta[5] = perturbed_theta[5] + d_hy
        return perturbed_theta

    def sample_gamma(self) -> float:
        """Samples random camera sensor gamma."""
        return float(torch.empty(1).uniform_(self.gamma_min, self.gamma_max).item())

    def apply_radiometric_eot(
        self,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Applies radiometric and atmospheric physical perturbations.
        x: (1, 3, H, W) in [0, 1]
        active_mask: (1, 1, H, W) optional region mask (e.g. patch region only)
        """
        device = x.device
        dtype = x.dtype

        # 1. Illumination variation: I = beta * I
        beta = torch.empty(1, device=device, dtype=dtype).uniform_(self.illum_min, self.illum_max)
        out = x * beta

        # 2. Atmospheric haze: I_haze = (1 - h) * I + h * A_air
        h = torch.empty(1, device=device, dtype=dtype).uniform_(0.0, self.haze_max)
        a_air = torch.tensor([0.80, 0.82, 0.85], device=device, dtype=dtype).view(1, 3, 1, 1)
        out = (1.0 - h) * out + h * a_air

        # 3. Defocus / atmospheric blur
        sigma_blur = float(torch.empty(1).uniform_(0.0, self.blur_max).item())
        if sigma_blur > 0.1:
            out = gaussian_blur_2d(out, sigma=sigma_blur, channels=3)

        # 4. Sensor additive Gaussian noise
        noise_sigma = float(torch.empty(1).uniform_(0.0, self.noise_max).item())
        if noise_sigma > 1e-4:
            noise = torch.randn_like(out) * noise_sigma
            out = out + noise

        # 5. Differentiable 8-bit quantization STE approximation
        out_clamped = torch.clamp(out, 0.0, 1.0)
        out_u8_approx = torch.round(out_clamped * 255.0) / 255.0
        out = out_u8_approx.detach() - out_clamped.detach() + out_clamped

        # If active_mask provided, blend only on active region to preserve background bit-identity
        if active_mask is not None:
            mask_3c = (active_mask > 0.0).repeat(1, 3, 1, 1)
            out = torch.where(mask_3c, out, x)

        return out
