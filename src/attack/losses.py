#!/usr/bin/env python3
"""
AE5510: Comprehensive Adversarial Camouflage Loss Formulation
=============================================================
Module: adv_camouflage/attack/losses.py

Implements:
  L = L_det + w_in * L_in + w_clear * L_clear + w_area * L_area + w_color * L_color + w_real * L_real

  1. L_det: Target detection suppression (SmoothMax confidence)
  2. L_in: Admissible region containment barrier loss
  3. L_clear: Clearance preservation penalty
  4. L_area: Compactness / patch scale regularization
  5. L_color: Binarization entropy penalty
  6. L_real: Total variation (TV) spatial coherence regularizer
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_camouflage.attack.placement import PatchPlacement
from adv_camouflage.attack.project import compute_containment_loss


def total_variation_loss(z: torch.Tensor) -> torch.Tensor:
    """Computes total variation loss on patch logits tensor (1, 1, H, W)."""
    diff_h = torch.abs(z[:, :, 1:, :] - z[:, :, :-1, :])
    diff_w = torch.abs(z[:, :, :, 1:] - z[:, :, :, :-1])
    return torch.mean(diff_h) + torch.mean(diff_w)


def binarization_entropy_loss(logits: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Penalizes soft probabilities near 0.5 to encourage distinct binary decisions:
    L_color = 4 * p * (1 - p), which is 1 at p=0.5 and 0 at p=0 or p=1.
    """
    p = torch.sigmoid(logits / tau)
    return torch.mean(4.0 * p * (1.0 - p))


class CompositeCamouflageLoss(nn.Module):
    def __init__(
        self,
        weight_in: float = 10.0,
        weight_clear: float = 0.5,
        weight_area: float = 2.0,
        weight_color: float = 0.5,
        weight_real: float = 0.1,
    ):
        super().__init__()
        self.w_in = weight_in
        self.w_clear = weight_clear
        self.w_area = weight_area
        self.w_color = weight_color
        self.w_real = weight_real

    def forward(
        self,
        loss_det: torch.Tensor,
        placement: PatchPlacement,
        D_outside: torch.Tensor,
        patch_logits: torch.Tensor,
        target_box_area_px: float,
        tau: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total loss and component breakdown.
        """
        # 1. Containment barrier loss
        loss_in = compute_containment_loss(placement, D_outside)

        # 2. Clearance penalty: penalizes offset tx, ty from peak clearance
        tx = placement.theta[0]
        ty = placement.theta[1]
        loss_clear = tx**2 + ty**2

        # 3. Compactness / Area loss: exp(log_s)
        log_s = placement.theta[2]
        loss_area = torch.exp(log_s)

        # 4. Binary entropy loss
        loss_color = binarization_entropy_loss(patch_logits, tau=tau)

        # 5. Total variation smoothness
        loss_real = total_variation_loss(patch_logits)

        # Total combined loss
        total_loss = (
            loss_det * 10.0
            + self.w_in * loss_in
            + self.w_clear * loss_clear
            + self.w_area * loss_area
            + self.w_color * loss_color
            + self.w_real * loss_real
        )

        breakdown = {
            "total_loss": total_loss.item(),
            "loss_det": loss_det.item(),
            "loss_in": loss_in.item(),
            "loss_clear": loss_clear.item(),
            "loss_area": loss_area.item(),
            "loss_color": loss_color.item(),
            "loss_real": loss_real.item(),
        }
        return total_loss, breakdown
