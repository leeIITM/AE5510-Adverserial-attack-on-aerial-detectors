#!/usr/bin/env python3
"""
AE5510: YOLO Detector Differentiable Wrapper
=============================================
Module: adv_camouflage/detector/yolo_wrapper.py

Provides:
  1. Frozen PyTorch inference on YOLO11 / YOLOv8 models
  2. Raw un-NMS head output extraction: bboxes (cx, cy, w, h) and class probabilities
  3. Differentiable target-anchor assignment via IoU and center containment
  4. Differentiable detection loss: L_det = SmoothMax(target_anchor_scores)
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


def box_iou_tensor(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise IoU between box1 (N, 4) and box2 (M, 4) in [x1, y1, x2, y2] format.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
    area2 = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / torch.clamp(union, min=1e-7)


class YOLODetectorWrapper(nn.Module):
    def __init__(self, weights_path: str, device: Optional[torch.device] = None):
        super().__init__()
        self.weights_path = weights_path
        self.yolo = YOLO(weights_path)
        self.model = self.yolo.model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.num_classes = len(self.yolo.names)
        self.class_names = self.yolo.names

    def forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs differentiable forward pass through the model.
        x: (B, 3, H, W) normalized to [0, 1]
        Returns:
          pred_boxes: (B, N, 4) in [x1, y1, x2, y2] pixel coordinates
          pred_scores: (B, N, C) class probabilities in [0, 1]
        """
        # Ensure spatial dimensions are multiples of 32 for FPN backbone
        H, W = x.shape[2:]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_in = x

        preds = self.model(x_in)[0]  # (B, 4 + C, N)
        # Transpose to (B, N, 4 + C)
        preds = preds.transpose(1, 2)

        cx, cy, w, h = preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, N, 4)
        pred_scores = preds[..., 4:]  # (B, N, C)
        return pred_boxes, pred_scores

    def compute_detection_loss(
        self,
        x: torch.Tensor,
        target_pixel_box: Tuple[int, int, int, int],
        target_class_id: int,
        iou_threshold: float = 0.35,
        temperature: float = 0.05,
        top_k: int = 20,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes differentiable detection suppression loss for a specific target.
        x: (1, 3, H, W)
        target_pixel_box: (bx1, by1, bx2, by2) in image pixel coordinates
        target_class_id: int
        Returns:
          loss_det: scalar torch.Tensor (differentiable w.r.t x)
          metrics: dict with max_conf, mean_conf, active_anchors
        """
        pred_boxes, pred_scores = self.forward_raw(x)
        boxes = pred_boxes[0]  # (N, 4)
        scores = pred_scores[0, :, target_class_id]  # (N,)

        tx1, ty1, tx2, ty2 = target_pixel_box
        tbox = torch.tensor([[tx1, ty1, tx2, ty2]], dtype=torch.float32, device=self.device)

        # 1. Target anchor selection: strictly select anchors predicting THIS target box (IoU >= threshold)
        ious = box_iou_tensor(boxes, tbox).squeeze(1)  # (N,)
        high_iou_mask = (ious >= iou_threshold)

        if high_iou_mask.sum() >= 3:
            cand_idx = torch.where(high_iou_mask)[0]
            cand_scores = scores[cand_idx]
            k_val = min(top_k, cand_scores.shape[0])
            active_scores, _ = torch.topk(cand_scores, k=k_val)
        else:
            # Fallback to top anchors with highest IoU with target box
            _, top_idx = torch.topk(ious, k=min(top_k, len(ious)))
            active_scores = scores[top_idx]

        # 2. SmoothMax loss approximation: tau * log(sum(exp(s / tau)))
        # Stable LogSumExp
        loss_det = temperature * torch.logsumexp(active_scores / temperature, dim=0)

        max_conf = torch.max(active_scores).item()
        mean_conf = torch.mean(active_scores).item()

        metrics = {
            "max_conf": max_conf,
            "mean_conf": mean_conf,
            "num_active_anchors": int(active_scores.shape[0]),
            "loss_det": loss_det.item(),
        }
        return loss_det, metrics
