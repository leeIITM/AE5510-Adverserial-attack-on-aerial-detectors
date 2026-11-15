#!/usr/bin/env python3
"""
Unit test: Gradient flow verification
Asserts that non-zero gradients flow through the entire pipeline:
  Detector -> Renderer -> Homography & Placement Theta -> Patch Logits
"""

import pytest
import torch
import torch.nn.functional as F

from adv_camouflage.detector.yolo_wrapper import YOLODetectorWrapper
from adv_camouflage.attack.render import PhysicalRadianceRenderer
from adv_camouflage.geometry.homography import build_homography_matrix


def test_end_to_end_gradient_flow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen detector
    detector = YOLODetectorWrapper("adv_camouflage/weights/yolo11s_best.pt", device=device)
    renderer = PhysicalRadianceRenderer().to(device)

    # Synthetic 640x640 clean image
    I_clean = torch.rand((1, 3, 640, 640), device=device, dtype=torch.float32)

    # Target box at center: (270, 270, 370, 370), class 0 (Aircraft)
    target_box = (270, 270, 370, 370)
    target_cls = 0

    # Hull mask inside target box
    yy, xx = torch.meshgrid(torch.arange(640, device=device), torch.arange(640, device=device), indexing="ij")
    M_hull = ((xx - 320) ** 2 + (yy - 320) ** 2 <= 40**2).float().unsqueeze(0).unsqueeze(0)

    # 1. Trainable patch logits z: shape (1, 1, 64, 64)
    z = torch.randn((1, 1, 64, 64), device=device, dtype=torch.float32, requires_grad=True)

    # Differentiable sigmoid pattern
    b = torch.sigmoid(z / 0.5)  # in (0, 1)
    c0 = torch.tensor([0.2, 0.2, 0.2], device=device).view(1, 3, 1, 1)
    c1 = torch.tensor([0.8, 0.8, 0.8], device=device).view(1, 3, 1, 1)
    patch_rgb = b * c1 + (1.0 - b) * c0

    # 2. Trainable placement parameters theta: (6,)
    # theta = [tx, ty, log_s, phi, hx, hy]
    theta = torch.tensor([0.0, 0.0, -0.5, 0.1, 0.02, -0.02], device=device, dtype=torch.float32, requires_grad=True)

    # 3. Homography
    H_fwd = build_homography_matrix(
        theta=theta,
        init_center_norm=(0.0, 0.0),
        box_scale_norm=(0.3, 0.3),
    )

    # 4. Render composite
    I_adv, alpha, _ = renderer.render(
        I_clean=I_clean,
        patch_rgb=patch_rgb,
        H_fwd=H_fwd,
        M_hull=M_hull,
        R_max=40.0,
    )

    # 5. Compute detection suppression loss
    loss_det, metrics = detector.compute_detection_loss(
        x=I_adv,
        target_pixel_box=target_box,
        target_class_id=target_cls,
    )

    assert torch.is_tensor(loss_det)
    assert not torch.isnan(loss_det)

    # 6. Backward pass
    loss_det.backward()

    # 7. Assertions on gradient flow
    assert z.grad is not None, "Gradient did not reach patch logits z!"
    z_grad_norm = torch.norm(z.grad).item()
    assert z_grad_norm > 0.0, f"Patch logits gradient norm is 0! ({z_grad_norm})"

    assert theta.grad is not None, "Gradient did not reach placement parameters theta!"
    theta_grad_norm = torch.norm(theta.grad).item()
    assert theta_grad_norm > 0.0, f"Placement theta gradient norm is 0! ({theta_grad_norm})"

    print(f"\n[Gradient Verification Success]:")
    print(f"  z.grad norm:     {z_grad_norm:.6f}")
    print(f"  theta.grad norm: {theta_grad_norm:.6f}")
    print(f"  loss_det value:  {loss_det.item():.6f}")
