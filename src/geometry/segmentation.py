#!/usr/bin/env python3
"""
AE5510: Target Silhouette Segmentation
======================================
Module: adv_camouflage/geometry/segmentation.py

Extracts precise object silhouette masks:
  1. Primary: YOLO11n-seg instance segmentation at conf=0.10
  2. IoU matching against the ground-truth target bounding box
  3. Fallback: GrabCut initialized from target bounding box (logged)
  4. Morphological erosion by 3px to eliminate ragged boundary pixels
"""

import logging
from typing import Tuple, Optional, Dict
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("adv_camouflage.segmentation")


class TargetSegmenter:
    def __init__(self, seg_model_path: str = "adv_camouflage/weights/yolo11n-seg.pt", conf: float = 0.10):
        self.conf = conf
        self.seg_model = None
        try:
            self.seg_model = YOLO(seg_model_path)
            # Suppress noisy ultralytics stdout
            self.seg_model.overrides["verbose"] = False
        except Exception as e:
            logger.warning(f"Could not load segmentation model from {seg_model_path}: {e}")

    def extract_silhouette(
        self,
        img_bgr: np.ndarray,
        target_pixel_box: Tuple[int, int, int, int],
        erode_kernel_size: int = 3,
    ) -> Tuple[np.ndarray, str]:
        """
        Extracts binary silhouette mask for the target object.
        target_pixel_box: (bx1, by1, bx2, by2) in integer pixels
        Returns:
          mask: uint8 array of shape (H, W) in {0, 255}
          method_used: 'yolo11n-seg' or 'grabcut_fallback'
        """
        H, W = img_bgr.shape[:2]
        bx1, by1, bx2, by2 = target_pixel_box
        box_area = max(1, (bx2 - bx1) * (by2 - by1))

        mask = None
        method_used = "none"

        # 1. Attempt YOLO11n-seg
        if self.seg_model is not None:
            try:
                # Run inference on image
                res = self.seg_model(img_bgr, conf=self.conf, verbose=False)[0]
                if res.masks is not None and len(res.masks) > 0 and res.boxes is not None:
                    best_iou = 0.0
                    best_idx = -1
                    for i, b in enumerate(res.boxes):
                        cx1, cy1, cx2, cy2 = map(int, b.xyxy.tolist()[0])
                        inter = max(0, min(bx2, cx2) - max(bx1, cx1)) * max(0, min(by2, cy2) - max(by1, cy1))
                        union = box_area + max(1, (cx2 - cx1) * (cy2 - cy1)) - inter
                        iou = inter / float(union)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i

                    if best_idx >= 0 and best_iou >= 0.15:
                        raw_mask = res.masks.data[best_idx].cpu().numpy()
                        # Resize raw mask back to image dimensions
                        mask_resized = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_LINEAR)
                        cand_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                        # Restrict to slightly padded bounding box to eliminate distant mask components
                        pad_x = int(0.15 * (bx2 - bx1))
                        pad_y = int(0.15 * (by2 - by1))
                        roi_mask = np.zeros_like(cand_mask)
                        rx1, ry1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
                        rx2, ry2 = min(W, bx2 + pad_x), min(H, by2 + pad_y)
                        roi_mask[ry1:ry2, rx1:rx2] = cand_mask[ry1:ry2, rx1:rx2]

                        if np.sum(roi_mask == 255) >= 20:
                            mask = roi_mask
                            method_used = "yolo11n-seg"
            except Exception as e:
                logger.warning(f"YOLO-Seg failed on sample: {e}")
                mask = None

        # 2. GrabCut Fallback
        if mask is None or np.sum(mask == 255) < 20:
            logger.info(f"YOLO-Seg missed target at {target_pixel_box}. Falling back to GrabCut.")
            method_used = "grabcut_fallback"
            mask = np.zeros((H, W), dtype=np.uint8)

            w_box = max(2, bx2 - bx1)
            h_box = max(2, by2 - by1)
            pad = max(10, int(0.20 * max(w_box, h_box)))
            cx1 = max(0, bx1 - pad)
            cy1 = max(0, by1 - pad)
            cx2 = min(W, bx2 + pad)
            cy2 = min(H, by2 + pad)

            crop = img_bgr[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]

            if cw > 4 and ch > 4:
                try:
                    gc_mask = np.zeros((ch, cw), dtype=np.uint8)
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    rx = bx1 - cx1
                    ry = by1 - cy1
                    rw = min(cw - rx - 1, w_box)
                    rh = min(ch - ry - 1, h_box)
                    rect = (max(1, rx), max(1, ry), max(2, rw), max(2, rh))
                    cv2.grabCut(crop, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                    fg_pixels = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                    mask[cy1:cy2, cx1:cx2] = fg_pixels
                except Exception as e:
                    logger.warning(f"GrabCut exception: {e}. Falling back to Otsu.")
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    mask[cy1:cy2, cx1:cx2] = thresh

        # 3. Apply 3px morphological erosion to eliminate ragged boundaries
        if erode_kernel_size > 0 and np.sum(mask == 255) > 50:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
            eroded = cv2.erode(mask, kernel, iterations=1)
            if np.sum(eroded == 255) >= 20:
                mask = eroded

        return mask, method_used
