#!/usr/bin/env python3
"""
Unit test: Containment and spillover verification
Asserts that after projection to the admissible region, the patch is 100% contained
within the silhouette mask, and measured spillover is exactly 0.000%.
"""

import pytest
import torch
import numpy as np

from adv_camouflage.attack.placement import PatchPlacement
from adv_camouflage.attack.project import measure_spillover, project_placement_to_admissible, compute_containment_loss
from adv_camouflage.attack.render import PhysicalRadianceRenderer
from adv_camouflage.geometry.distance import DistanceAnalyzer


def test_hard_containment_and_zero_spillover():
    H, W = 300, 300
    # Create target silhouette: circle of radius 50 at (150, 150)
    yy, xx = np.ogrid[:H, :W]
    mask_np = ((xx - 150) ** 2 + (yy - 150) ** 2 <= 50**2).astype(np.uint8) * 255

    dist_analyzer = DistanceAnalyzer()
    dist_res = dist_analyzer.analyze_silhouette(mask_np)

    # Initialize a placement that is intentionally too large (scale fraction 0.95)
    placement = PatchPlacement(
        init_center=(150.0, 150.0),
        base_radius=float(dist_res["R_max"]),
        canvas_hw=(H, W),
        target_box_wh=(100.0, 100.0),
        initial_scale_fraction=0.95,
    )

    # Project to admissible region
    proj_res = project_placement_to_admissible(
        placement=placement,
        admissible_mask=dist_res["admissible_mask"],
        D_inside=dist_res["D_inside"],
    )

    assert proj_res["converged"], "Projection failed to converge!"

    # Now render the warped patch footprint and measure spillover
    renderer = PhysicalRadianceRenderer()
    patch_rgb = torch.ones((1, 3, 64, 64), dtype=torch.float32)
    I_clean = torch.zeros((1, 3, H, W), dtype=torch.float32)
    M_hull_tensor = torch.from_numpy(mask_np > 128).float().unsqueeze(0).unsqueeze(0)

    H_fwd = placement.get_homography()
    _, alpha, warped_patch_srgb = renderer.render(
        I_clean=I_clean,
        patch_rgb=patch_rgb,
        H_fwd=H_fwd,
        M_hull=torch.ones_like(M_hull_tensor),  # unmasked hull to observe true physical sticker footprint
        R_max=float(dist_res["R_max"]),
    )

    spill_stats = measure_spillover(warped_alpha=alpha, M_hull=M_hull_tensor, threshold=0.01)
    spill_rate = spill_stats["spillover_rate"]

    print(f"\n[Hard Containment Check]:")
    print(f"  Footprint pixels:  {spill_stats['footprint_pixels']}")
    print(f"  Spillover pixels:  {spill_stats['spillover_pixels']}")
    print(f"  Spillover rate:    {spill_rate * 100.0:.3f}%")

    assert spill_rate == 0.0, f"Spillover rate was {spill_rate * 100.0:.3f}%, expected 0.000%!"


def test_containment_loss_differentiable():
    H, W = 200, 200
    # D_outside: Euclidean distance from centered circle of radius 40
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist_from_center = torch.sqrt((xx - 100.0) ** 2 + (yy - 100.0) ** 2)
    D_outside = torch.clamp(dist_from_center - 40.0, min=0.0).unsqueeze(0).unsqueeze(0)

    placement = PatchPlacement(
        init_center=(100.0, 100.0),
        base_radius=40.0,
        canvas_hw=(H, W),
        target_box_wh=(80.0, 80.0),
        initial_scale_fraction=0.8,
    )

    loss_cont = compute_containment_loss(placement, D_outside)
    loss_cont.backward()

    assert placement.theta.grad is not None
    grad_norm = torch.norm(placement.theta.grad).item()
    print(f"Containment loss gradient norm: {grad_norm:.6f}")
    assert grad_norm >= 0.0
