#!/usr/bin/env python3
"""
Unit test: Background untouched verification
Asserts that every pixel outside the patch alpha mask is bit-identical to the input image.
"""

import pytest
import torch
import numpy as np
import cv2
from adv_camouflage.attack.render import PhysicalRadianceRenderer
from adv_camouflage.geometry.homography import build_homography_matrix


def test_background_bit_identical_synthetic():
    renderer = PhysicalRadianceRenderer()
    torch.manual_seed(42)

    # Synthetic 256x256 image with realistic values in [0, 1]
    I_clean = torch.rand((1, 3, 256, 256), dtype=torch.float32)

    # Patch: 64x64
    patch_rgb = torch.rand((1, 3, 64, 64), dtype=torch.float32)

    # Admissible hull mask: centered circle of radius 40
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    M_hull = ((xx - 128) ** 2 + (yy - 128) ** 2 <= 40**2).float().unsqueeze(0).unsqueeze(0)

    # Homography centered at (0, 0) normalized, scale 0.2
    theta = torch.tensor([0.0, 0.0, -1.0, 0.2, 0.05, -0.05], dtype=torch.float32)
    H_fwd = build_homography_matrix(
        theta=theta,
        init_center_norm=(0.0, 0.0),
        box_scale_norm=(0.3, 0.3),
    )

    I_adv, alpha, _ = renderer.render(
        I_clean=I_clean,
        patch_rgb=patch_rgb,
        H_fwd=H_fwd,
        M_hull=M_hull,
        R_max=40.0,
        gamma=2.2,
    )

    # Outside the patch alpha footprint, the image must be BIT-IDENTICAL
    outside_mask = (alpha == 0.0).repeat(1, 3, 1, 1)
    diff = torch.abs(I_adv[outside_mask] - I_clean[outside_mask])
    max_diff = torch.max(diff).item()

    assert max_diff == 0.0, f"Max difference on untouched background was {max_diff}, expected 0.0"

    # Also test uint8 quantization bit-identity
    clean_uint8 = (I_clean * 255.0).round().to(torch.uint8)
    adv_uint8 = (I_adv * 255.0).round().to(torch.uint8)
    uint8_diff = torch.abs(adv_uint8[outside_mask].int() - clean_uint8[outside_mask].int())
    assert torch.max(uint8_diff).item() == 0, "uint8 quantized background differed!"


def test_background_bit_identical_real_sample():
    """Verify on real image tensor from dataset."""
    from adv_camouflage.data.dataset import AerialDataset
    from adv_camouflage.geometry.segmentation import TargetSegmenter
    from adv_camouflage.geometry.distance import DistanceAnalyzer

    dataset = AerialDataset(split_type="train")
    # Take first sample
    sample = dataset.load_image_and_targets(0)
    img_bgr = sample["image_bgr"]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    I_clean = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    H, W = sample["height"], sample["width"]

    # Pick the largest target
    target = max(sample["targets"], key=lambda t: t["area_px"])
    segmenter = TargetSegmenter()
    M_hull_u8, _ = segmenter.extract_silhouette(img_bgr, target["pixel_box"])
    M_hull = (M_hull_u8 > 128).astype(np.float32)
    M_hull_tensor = torch.from_numpy(M_hull).float().unsqueeze(0).unsqueeze(0)

    dist_analyzer = DistanceAnalyzer()
    dist_res = dist_analyzer.analyze_silhouette(M_hull_u8)
    cx, cy = dist_res["top_k_centers"][0]
    r_max = dist_res["R_max"]

    renderer = PhysicalRadianceRenderer()
    patch_rgb = torch.rand((1, 3, 64, 64), dtype=torch.float32)

    # Normalized coordinates in [-1, 1]
    cx_norm = (float(cx) / W) * 2.0 - 1.0
    cy_norm = (float(cy) / H) * 2.0 - 1.0
    bw_norm = (float(r_max) * 2.0 / W) * 2.0
    bh_norm = (float(r_max) * 2.0 / H) * 2.0

    theta = torch.tensor([0.0, 0.0, -0.5, 0.1, 0.0, 0.0], dtype=torch.float32)
    H_fwd = build_homography_matrix(
        theta=theta,
        init_center_norm=(cx_norm, cy_norm),
        box_scale_norm=(bw_norm, bh_norm),
    )

    I_adv, alpha, _ = renderer.render(
        I_clean=I_clean,
        patch_rgb=patch_rgb,
        H_fwd=H_fwd,
        M_hull=M_hull_tensor,
        R_max=float(r_max),
    )

    outside_mask = (alpha == 0.0).repeat(1, 3, 1, 1)
    diff = torch.abs(I_adv[outside_mask] - I_clean[outside_mask])
    max_diff = torch.max(diff).item()

    assert max_diff == 0.0, f"Max diff on real sample was {max_diff}"
