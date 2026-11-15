#!/usr/bin/env python3
"""
AE5510: Physical Radiance Renderer and Compositor
=================================================
Module: adv_camouflage/attack/render.py

Implements physical radiance compositing and sensor simulation:
  1. sRGB decode to linear radiance (I_lin = I_srgb ** 2.2)
  2. Local shading extraction from vehicle luminance: L = GaussianBlur(Y, sigma=0.15*R_max)
  3. Modulates linear patch radiance by normalized local shading
  4. Feathered alpha boundary: GaussianBlur(warped_mask, sigma=1.5px) * M_hull
  5. Linear radiance compositing: I_adv_lin = (1 - alpha) * I_lin + alpha * P_lin
  6. Camera sensor radiometry: I_adv = clamp(I_adv_lin, 0, 1) ** (1 / gamma)
  7. Hard Background Guarantee: I_adv[~mask_union] = I_clean[~mask_union] (bit-identical)
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_camouflage.geometry.homography import warp_patch_to_canvas


def get_gaussian_kernel2d(
    sigma: float, channels: int = 1, device=None, dtype=torch.float32
) -> Tuple[torch.Tensor, int]:
    """Builds a normalized 2D Gaussian kernel for differentiable filtering."""
    sigma = max(0.1, float(sigma))
    radius = int(math.ceil(2.0 * sigma))
    kernel_size = 2 * radius + 1

    coords = torch.arange(kernel_size, device=device, dtype=dtype) - radius
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel, radius


def gaussian_blur_2d(
    x: torch.Tensor, sigma: float, channels: int = 1
) -> torch.Tensor:
    """Applies differentiable 2D Gaussian blur with reflection padding."""
    if sigma <= 0.05:
        return x
    kernel, pad = get_gaussian_kernel2d(sigma, channels, device=x.device, dtype=x.dtype)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_padded, kernel, groups=channels)


class PhysicalRadianceRenderer(nn.Module):
    def __init__(
        self,
        srgb_decode_gamma: float = 2.2,
        shading_sigma_fraction: float = 0.15,
        shading_clip_min: float = 0.60,
        shading_clip_max: float = 1.40,
        feather_sigma_px: float = 1.50,
        sensor_gamma_eval: float = 2.0,
    ):
        super().__init__()
        self.decode_gamma = srgb_decode_gamma
        self.shading_sigma_fraction = shading_sigma_fraction
        self.shading_clip_min = shading_clip_min
        self.shading_clip_max = 1.40  # [0.6, 1.4]
        self.feather_sigma_px = feather_sigma_px
        self.sensor_gamma_eval = sensor_gamma_eval

    def srgb_to_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Converts sRGB [0, 1] to linear radiance."""
        return torch.pow(torch.clamp(x, 1e-6, 1.0), self.decode_gamma)

    def linear_to_sensor(self, x_lin: torch.Tensor, gamma: float) -> torch.Tensor:
        """Converts linear radiance to camera sensor output with exponent 1 / gamma."""
        return torch.pow(torch.clamp(x_lin, 1e-6, 1.0), 1.0 / gamma)

    def compute_local_shading(
        self,
        I_lin: torch.Tensor,
        M_hull: torch.Tensor,
        R_max: float,
    ) -> torch.Tensor:
        """
        Extracts local surface shading map restricted to vehicle hull M_hull:
        Y = luminance(I_lin)
        L = GaussianBlur(Y, sigma = 0.15 * R_max)
        shade = L / mean(L over M), clipped to [0.6, 1.4]
        """
        # Linear relative luminance
        Y = 0.2126 * I_lin[:, 0:1] + 0.7152 * I_lin[:, 1:2] + 0.0722 * I_lin[:, 2:3]

        sigma_shading = max(1.0, self.shading_sigma_fraction * R_max)
        L = gaussian_blur_2d(Y, sigma=sigma_shading, channels=1)

        mask_active = (M_hull > 0.5)
        if mask_active.any():
            L_mean = torch.mean(L[mask_active])
        else:
            L_mean = torch.mean(L)

        shade = L / (L_mean + 1e-6)
        shade = torch.clamp(shade, self.shading_clip_min, self.shading_clip_max)
        return shade

    def render(
        self,
        I_clean: torch.Tensor,
        patch_rgb: torch.Tensor,
        H_fwd: torch.Tensor,
        M_hull: torch.Tensor,
        R_max: float,
        gamma: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        I_clean: (1, 3, H, W) normalized [0, 1] sRGB
        patch_rgb: (1, 3, h_p, w_p) normalized [0, 1] sRGB
        H_fwd: (3, 3) forward homography matrix
        M_hull: (1, 1, H, W) binary vehicle hull mask in {0, 1}
        R_max: inscribed radius peak clearance
        gamma: camera sensor gamma (defaults to sensor_gamma_eval)
        Returns:
          I_adv: (1, 3, H, W) composite image with BIT-IDENTICAL background
          alpha: (1, 1, H, W) feathered, hull-confined alpha mask
          patch_warped_rgb: (1, 3, H, W) warped patch RGB
        """
        H, W = I_clean.shape[2:]
        device = I_clean.device
        dtype = I_clean.dtype
        sensor_gamma = self.sensor_gamma_eval if gamma is None else gamma

        # 1. Convert clean image to linear radiance
        I_lin = self.srgb_to_linear(I_clean)

        # 2. Canonical patch alpha mask: footprint is 1 inside canonical coordinates
        h_p, w_p = patch_rgb.shape[2:]
        base_patch_alpha = torch.ones((1, 1, h_p, w_p), device=device, dtype=dtype)

        # 3. Differentiable perspective warp to canvas
        warped_patch_srgb, warped_alpha = warp_patch_to_canvas(
            patch_rgb, base_patch_alpha, H_fwd, (H, W)
        )

        # 4. Local surface shading in linear radiance
        shade = self.compute_local_shading(I_lin, M_hull, R_max)
        warped_patch_lin = self.srgb_to_linear(warped_patch_srgb)
        P_lin = warped_patch_lin * shade

        # 5. Feathered alpha boundary confined strictly to M_hull
        alpha_feathered = gaussian_blur_2d(warped_alpha, sigma=self.feather_sigma_px, channels=1)
        alpha = torch.clamp(alpha_feathered, 0.0, 1.0) * M_hull

        # 6. Composite in linear radiance
        I_adv_lin = (1.0 - alpha) * I_lin + alpha * P_lin

        # 7. Apply camera sensor gamma
        I_adv_sensor = self.linear_to_sensor(I_adv_lin, gamma=sensor_gamma)

        # 8. BACKGROUND GUARANTEE:
        # Every pixel with alpha == 0 must be BIT-IDENTICAL to I_clean!
        mask_union = (alpha > 0.0).repeat(1, 3, 1, 1)
        I_adv = torch.where(mask_union, I_adv_sensor, I_clean)

        return I_adv, alpha, warped_patch_srgb
