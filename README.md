# Physically Grounded Binary Adversarial Camouflage for Real-Time Aerial Object Detectors Under Flight Dynamics

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE Format](https://img.shields.io/badge/IEEE-Conference%20Paper-00629B.svg)](paper/AE5510_Problem1_Paper.pdf)

**Author:** [leeIITM](https://github.com/leeIITM)  
**Department of Aerospace Engineering, IIT Madras**  
**Course Project:** AE5510 Machine Learning Applications — Problem 1

---

## Overview

Uncrewed aerial reconnaissance vehicles (UAVs) increasingly deploy single-stage deep convolutional object detectors (YOLO11 and YOLOv8) to rapidly detect and localize high-value tactical assets, including parked aircraft, military transport vehicles, tanks, and command infrastructure. While deep detectors achieve high nominal accuracy, they are vulnerable to adversarial physical camouflage.

Prior physical patch attacks typically assume unconstrained, multi-spectral 24-bit RGB printing placed indiscriminately across scenes. In real-world aerial operations, such attacks suffer from:
1. **Unrealistic Chromatic Complexity:** Printer gamut clipping and atmospheric attenuation dilute continuous 24-bit RGB gradient noise.
2. **Unconstrained Background Spillover:** Slapping rectangular patches over bounding boxes alters surrounding terrain, violating operational stealth.
3. **Aerial Dynamic Fragility:** Digital patches fail under varying solar zenith angles, atmospheric haze, platform vibration, and radiometric sensor gamma curves.

This repository implements a **physically realizable, background-aware binary adversarial camouflage framework**. Our approach jointly optimizes a discrete two-tone vinyl decal pattern and a 6-DoF conformal homography warp strictly confined within the target asset silhouette. Using a differentiable Straight-Through Estimator (STE) and post-convergence hard geometric projection, our system guarantees **0.000% background spillover** and **bit-level background pixel identity** outside the target silhouette ($\max |\Delta| = 0.000000$).

---

## Framework Architecture

```
[ Aerial Image + Bounding Box ]
              │
      ┌───────┴────────────────────────┐
      ▼                                ▼
[ Target Silhouette ]        [ Background Context Ring ]
  (YOLO11n-seg + GrabCut)      (15px ambient dilation)
      │                                │
      ▼                                ▼
[ Distance Transform Map ]   [ CIELAB 2-Tone Palette ]
  (Peak Clearance Center)      (ΔL* ≥ 25.0, ΔE* ≤ 30.0)
      │                                │
      └───────┬────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────┐
   │         Joint Optimization Engine            │
   │  • Pattern Logits z ∈ R^(48x48) via STE      │
   │  • 6-DoF Conformal Homography Pose θ         │
   │  • Expectation over Transformations (EoT)    │
   │  • Differentiable Surrogate Loss (YOLO11s)   │
   └──────────────────────────────────────────────┘
              │
              ▼
   [ Hard Geometric Projection Barrier ]
     (Iterative shrinkage to 0.000% spillover)
              │
              ▼
   [ Photometric Sensor Radiance Pipeline ]
     (Linear shading, feathered alpha, sensor gamma)
              │
              ▼
   [ Deployable 2-Tone Vinyl Decal + Coordinates ]
```

---

## Key Verified Results

All metrics below are generated directly from the reproducible audit evaluation on the official 16-scene tactical aerial test set (132 ground-truth targets):

### 1. Physical Containment & Integrity Verification
| Quality Metric | Benchmark Requirement | Measured Result | Audit Status |
| :--- | :---: | :---: | :---: |
| **Mean Background Spillover** | $0.000\%$ | **0.000%** | **VERIFIED (PASS)** |
| **Maximum Observed Spillover** | $0.000\%$ | **0.000%** | **VERIFIED (PASS)** |
| **Zero-Spillover Target Rate** | $100.0\%$ | **100.0% (132/132)** | **VERIFIED (PASS)** |
| **Max Abs Background Difference ($\Delta$)** | $0.000000$ | **0.000000** | **VERIFIED (PASS)** |
| **Unique Palette Colors** | Exactly 2 | **2 Tones (Vinyl)** | **VERIFIED (PASS)** |

### 2. Binary vs. Continuous Camouflage Head-to-Head ASR
Evaluated across confirmed clean detections under radiometric sensor gamma $\gamma = 2.2$ and physical color drift:
| Target Class | Clean Detections | Binary Blinded | Binary ASR | Continuous Blinded | Continuous ASR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Aircraft** | 7 | 3 | 42.86% | 3 | 42.86% |
| **Building** | 20 | 18 | **90.00%** | 19 | **95.00%** |
| **Military Truck** | 1 | 1 | 100.00% | 1 | 100.00% |
| **Tank** | 2 | 1 | 50.00% | 1 | 50.00% |
| **Overall Benchmark** | **30** | **23** | **76.67%** | **24** | **80.00%** |

*Takeaway:* Two-tone vinyl decals match continuous unconstrained RGB gradient attacks (**76.67% vs. 80.00% ASR**) while being 100% field-printable.

### 3. Component Ablation Study (YOLO11s)
| Configuration | Detected Targets | Detection Rate | Mean Confidence | Attack Success Rate (ASR) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Clean (No Patch)** | 35 / 77 | 45.45% | 0.3114 | 0.00% |
| **Naive Slapped Sticker** | 15 / 77 | 19.48% | 0.1062 | 57.14% |
| **Spatial-Only (No Pattern)** | 29 / 77 | 37.66% | 0.2626 | 17.14% |
| **Texture-Only (Fixed Center)** | 20 / 77 | 25.97% | 0.1730 | 42.86% |
| **Proposed Full Framework** | **2 / 77** | **2.60%** | **0.0173** | **94.29%** |

### 4. Cross-Model Transferability (132 Targets)
| Model | Attack Role | Clean mAP50 | Adv mAP50 | Overall Drop | Building AP50 Drop |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLO11s** | Whitebox Source | 43.79% | 37.27% | **-6.52%** | **-11.50%** |
| **YOLO11m** | Blackbox Scaled | 38.43% | 37.26% | **-1.17%** | **-11.85%** |
| **YOLOv8s** | Blackbox Family | 45.07% | 44.03% | **-1.04%** | **-15.17%** |

---

## Repository Structure

```
.
├── configs/
│   └── default.yaml             # Master configuration file (relative paths)
├── adv_camouflage/              # Core modular library
│   ├── attack/                  # STE pattern generator, homography pose, EOT, losses, projection
│   ├── detector/                # Differentiable YOLO wrapper & evaluation metrics
│   ├── geometry/                # Segmentation, Euclidean distance transform, CIELAB palette
│   └── utils/                   # Label sanitization & 4-corner affine transformation
├── dataset/
│   └── Testing Data/            # Official 16 testing scenes + audit-corrected labels
├── weights/                     # Fine-tuned PyTorch model checkpoints (<40MB each)
│   ├── yolo11s_best.pt          # Primary whitebox model (69.88% mAP50)
│   ├── yolo11m_best.pt          # Blackbox scaled model (66.51% mAP50)
│   ├── yolov8s_best.pt          # Blackbox family model (59.36% mAP50)
│   └── yolo11n-seg.pt           # Semantic segmentation model
├── outputs/
│   ├── results/                 # Verified benchmark JSON files
│   └── figures/                 # All high-resolution analytical figures
├── paper/                       # Complete IEEE Conference Publication
│   ├── AE5510_Problem1_Paper.tex # 2-Column LaTeX paper with all figures embedded
│   ├── refs.bib                 # BibTeX references
│   ├── AE5510_Problem1_Paper.pdf # Compiled 7-page PDF
│   └── figures/                 # Embedded graphic assets
├── presentation/
│   └── AE5510_Problem1_Beamer.tex # Project defense presentation slides
├── scripts/                     # 1-Line reproduction scripts
│   ├── run_attack.py            # Run camouflage attack pipeline on test scene
│   ├── verify_containment.py    # Assert 0.000% spillover & background invariance
│   ├── evaluate_transfer.py     # Reproduce cross-model transfer benchmark
│   ├── evaluate_binary_vs_continuous.py # Reproduce head-to-head ASR benchmark
│   └── compile_paper.py         # Re-compile paper PDF
├── tests/                       # Automated test suite
│   ├── test_binary_pattern.py   # Assert 2 unique colors & STE gradient flow
│   ├── test_containment.py      # Assert hard containment projection
│   └── test_background_untouched.py # Assert zero background modification
├── requirements.txt             # Python dependencies
├── setup.py                     # Local package installer
└── README.md
```

---

## Quick Start & Reproduction

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/leeIITM/AE5510-Adversarial-Camouflage.git
cd AE5510-Adversarial-Camouflage
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Physical Containment & Background Identity
Verifies that all 132 test targets strictly maintain 0.000% spillover and background invariance:
```bash
python scripts/verify_containment.py
```
*Expected Output:*
```
========================================================================
PHYSICAL CONTAINMENT & BACKGROUND INVARIANCE AUDIT
========================================================================
[1] Spillover Evaluation (N=132 targets):
    - Mean Spillover: 0.0000%  (Required: 0.000%)
    - Max Spillover:  0.0000%  (Required: 0.000%)
    - Zero-Spillover Target Rate: 100.0% (Required: 100.0%)
    --> PASS: Hard containment projection strictly verified!

[2] Background Pixel Invariance (N=16 scenes):
    - All Bit-Identical: True
    --> PASS: Bit-level background identity strictly preserved (Δ = 0.000000)!

[3] Palette Discrete Cardinality:
    - Allowed Colors: Exactly 2 (Two-Tone Vinyl Decal)
    --> PASS: Design uses discrete binary palette {c0, c1}!
========================================================================
ALL AUDIT CHECKS PASSED: 100% EMPIRICALLY COMPLIANT
========================================================================
```

### 3. Reproduce Cross-Model Transferability
Evaluates whitebox YOLO11s patches against blackbox YOLO11m and YOLOv8s detectors:
```bash
python scripts/evaluate_transfer.py
```

### 4. Reproduce Binary vs. Continuous Head-to-Head Comparison
```bash
python scripts/evaluate_binary_vs_continuous.py
```

### 5. Run Camouflage Optimization on a Sample Scene
Executes joint placement and pattern optimization on a sample test scene:
```bash
python scripts/run_attack.py
```
Output adversarial images are saved to `outputs/adversarial_images/`.

### 6. Compile IEEE Conference Paper PDF
```bash
python scripts/compile_paper.py
```

### 7. Run Unit Tests
```bash
pytest tests/ -v
```

---

## IEEE Conference Paper & Citation

The complete publication draft is available at [`paper/AE5510_Problem1_Paper.pdf`](paper/AE5510_Problem1_Paper.pdf). To cite this work:

```bibtex
@inproceedings{lee2026adversarial,
  title={Physically Grounded Binary Adversarial Camouflage for Real-Time Aerial Object Detectors Under Flight Dynamics},
  author={leeIITM},
  booktitle={Department of Aerospace Engineering, AE5510 Machine Learning Applications Technical Report},
  year={2026},
  url={https://github.com/leeIITM/AE5510-Adversarial-Camouflage}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
