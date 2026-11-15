#!/usr/bin/env python3
"""
Unit test: Binary pattern purity verification
Asserts that the exported patch design consists of EXACTLY two colours c0 and c1,
with zero intermediate colors, and that STE gradients flow during hard forward mode.
"""

import pytest
import torch
import numpy as np
from adv_camouflage.attack.pattern import BinaryPatternGenerator


def test_binary_pattern_exact_two_colors():
    generator = BinaryPatternGenerator(height=64, width=64)
    c0 = torch.tensor([0.25, 0.35, 0.45], dtype=torch.float32)
    c1 = torch.tensor([0.75, 0.85, 0.95], dtype=torch.float32)

    # 1. Test exported design image
    design_u8 = generator.get_discrete_design_image(c0, c1)
    assert design_u8.shape == (64, 64, 3)

    # Count unique RGB 3-tuples
    pixels = design_u8.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    # There must be EXACTLY 2 unique colors in the design file
    assert len(unique_colors) == 2, f"Expected exactly 2 colors, found {len(unique_colors)}: {unique_colors}"

    # Verify colors match c0 and c1 in 8-bit
    c0_u8 = (c0.numpy() * 255.0).round().astype(np.uint8)
    c1_u8 = (c1.numpy() * 255.0).round().astype(np.uint8)

    assert any(np.array_equal(u, c0_u8) for u in unique_colors), "c0 color missing in design"
    assert any(np.array_equal(u, c1_u8) for u in unique_colors), "c1 color missing in design"


def test_ste_gradients_flow_during_hard_mode():
    generator = BinaryPatternGenerator(height=32, width=32)
    c0 = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32)
    c1 = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32)

    # Forward in forced hard mode (STE)
    patch_rgb, b = generator(c0, c1, force_hard=True)

    # Verify b is strictly discrete in forward pass
    assert torch.all((b == 0.0) | (b == 1.0)), "STE forward pass contained non-binary values!"

    # Loss computation
    dummy_loss = (patch_rgb * 2.0).sum()
    dummy_loss.backward()

    # Verify gradients passed through STE to continuous logits
    assert generator.logits.grad is not None, "Gradients failed to flow through STE!"
    grad_norm = torch.norm(generator.logits.grad).item()
    assert grad_norm > 0.0, f"STE gradient norm was 0! ({grad_norm})"
