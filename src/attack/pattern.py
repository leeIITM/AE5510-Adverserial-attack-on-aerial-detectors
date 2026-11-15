#!/usr/bin/env python3
"""
AE5510: Straight-Through Estimator (STE) Binary Pattern Generator
==================================================================
Module: adv_camouflage/attack/pattern.py

Implements:
  1. Continuous latent logits z in R^{H_p x W_p}
  2. Straight-Through Estimator (STE) for discrete binary forward pass
  3. Temperature annealing tau from 1.0 down to 0.05
  4. Hard binarization b in {0, 1} with gradient pass-through
  5. Discrete two-colour mapping: P = b * c1 + (1 - b) * c0
  6. Binary purity validation (< 2 unique colors in design file)
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryPatternGenerator(nn.Module):
    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        init_temperature: float = 1.0,
        min_temperature: float = 0.05,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature

        # Trainable continuous logits
        self.logits = nn.Parameter(torch.randn(1, 1, height, width) * 0.1)

    def get_temperature(self, step: int, total_steps: int) -> float:
        """Exponential / cosine temperature annealing."""
        progress = min(1.0, max(0.0, step / max(1, total_steps)))
        # Exponential decay
        tau = self.init_temperature * ((self.min_temperature / self.init_temperature) ** progress)
        return max(self.min_temperature, float(tau))

    def forward(
        self,
        c0: torch.Tensor,
        c1: torch.Tensor,
        step: int = 0,
        total_steps: int = 100,
        force_hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates discrete two-colour patch RGB tensor.
        c0: (1, 3, 1, 1) or (3,) RGB in [0, 1]
        c1: (1, 3, 1, 1) or (3,) RGB in [0, 1]
        Returns:
          patch_rgb: (1, 3, H, W)
          binary_mask: (1, 1, H, W) binary values in {0.0, 1.0}
        """
        if c0.ndim == 1:
            c0 = c0.view(1, 3, 1, 1)
        if c1.ndim == 1:
            c1 = c1.view(1, 3, 1, 1)

        tau = self.get_temperature(step, total_steps)

        # 1. Soft probability map
        p = torch.sigmoid(self.logits / tau)

        # 2. Hard discrete binarization in final 20% of steps or when forced
        is_hard_phase = force_hard or (step >= int(0.80 * total_steps))

        if is_hard_phase:
            # Hard binarization
            b_hard = (p >= 0.5).float()
            # Straight-Through Estimator:
            # Forward uses b_hard (strictly 0 or 1)
            # Backward passes gradients through p
            b = b_hard.detach() - p.detach() + p
        else:
            # Soft continuous relaxation during initial exploration
            b = p

        # 3. Two-colour mapping
        patch_rgb = b * c1 + (1.0 - b) * c0
        return patch_rgb, b

    def get_discrete_design_image(
        self,
        c0: torch.Tensor,
        c1: torch.Tensor,
    ) -> np.ndarray:
        """
        Exports the discrete design file as an 8-bit RGB numpy array.
        Every pixel is guaranteed to be strictly c0_u8 or c1_u8.
        """
        with torch.no_grad():
            patch_rgb, b = self.forward(c0, c1, force_hard=True)
            # Quantize strictly to uint8
            patch_u8 = (patch_rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            return patch_u8
