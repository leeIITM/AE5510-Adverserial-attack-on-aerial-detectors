# Binary Adversarial Camouflage for Aerial Object Detectors

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [leeIITM](https://github.com/leeIITM)
**Department of Aerospace Engineering, IIT Madras**
**Course project:** AE5510 Machine Learning Applications, Problem 1

---

## What this project is about

Drones increasingly use object detectors like YOLO to spot aircraft, vehicles, tanks and buildings from the air. These detectors can be fooled by specially designed patterns, called adversarial patches. This project asks a practical question:

> Can a simple two-colour sticker, placed only on the object itself, stop a detector from seeing it?

Most earlier patch attacks have three problems when used on aerial images:

1. They use full-colour patterns that are hard to print accurately.
2. They are placed as rectangles, so they spill onto the ground around the object.
3. They are not tested against things like haze, lighting changes and camera blur.

Our approach fixes these by using only **two colours** (like cut vinyl), keeping the sticker **inside the object's outline**, and training it under **flight-like conditions**.

---

## How it works

1. **Find the object outline** with a segmentation model (YOLO11n-seg, with GrabCut as backup).
2. **Pick two colours** that blend with the area around the object but still contrast with each other.
3. **Train the pattern and its position together.** The pattern is a 48x48 black-and-white grid, learned with a Straight-Through Estimator. Its position, size, rotation and shear are learned at the same time.
4. **Keep it inside the object.** If any part of the sticker lands outside the outline, it is shrunk and moved inward.
5. **Simulate flight.** During training the image is randomly rotated, zoomed, hazed, darkened or brightened, blurred and given sensor noise, so the pattern is not tied to one perfect view.

The detector used for training (YOLO11s) is the "whitebox" model. YOLO11m and YOLOv8s are used as unseen "blackbox" models to test transfer.

---

## Main results

Tested on 16 aerial scenes with 132 labelled targets.

### Two-colour vs. full-colour patch

Attack success rate (ASR) is the share of originally detected targets that the detector no longer finds.

| Class | Clean detections | Binary ASR | Continuous ASR |
| :--- | :---: | :---: | :---: |
| Aircraft | 7 | 42.86% | 42.86% |
| Building | 20 | 90.00% | 95.00% |
| Military Truck | 1 | 100.00% | 100.00% |
| Tank | 2 | 50.00% | 50.00% |
| **Overall** | **30** | **76.67%** | **80.00%** |

The two-colour patch is almost as effective as the full-colour one, and it is much easier to print. Note that only 30 targets were detected on clean images, so these numbers are based on a small sample.

### What each part contributes (YOLO11s)

| Setup | Detected | Mean confidence | ASR |
| :--- | :---: | :---: | :---: |
| No patch | 35 / 77 | 0.3114 | 0.00% |
| Naive sticker | 15 / 77 | 0.1062 | 57.14% |
| Placement only | 29 / 77 | 0.2626 | 17.14% |
| Pattern only | 20 / 77 | 0.1730 | 42.86% |
| **Full method** | **2 / 77** | **0.0173** | **94.29%** |

Optimising the pattern and the position together works far better than either one alone.

### Transfer to other detectors

| Model | Role | Clean mAP50 | Attacked mAP50 | Building AP50 drop |
| :--- | :--- | :---: | :---: | :---: |
| YOLO11s | Source | 43.79% | 37.27% | -11.50% |
| YOLO11m | Unseen | 38.43% | 37.26% | -11.85% |
| YOLOv8s | Unseen | 45.07% | 44.03% | -15.17% |

Overall mAP barely changes on the unseen models, but buildings are clearly affected.

---

## Repository structure

```
.
├── configs/
│   └── default.yaml              # Main settings
├── adv_camouflage/               # Core code
│   ├── attack/                   # Pattern, placement, EOT, losses, projection
│   ├── detector/                 # YOLO wrapper and metrics
│   ├── geometry/                 # Outline, distance transform, colour palette
│   └── utils/                    # Label fixing and affine transforms
├── dataset/
│   └── Testing Data/             # 16 test scenes with corrected labels
├── weights/                      # Fine-tuned model checkpoints
│   ├── yolo11s_best.pt
│   ├── yolo11m_best.pt
│   ├── yolov8s_best.pt
│   └── yolo11n-seg.pt
├── outputs/
│   ├── results/                  # Result files (JSON)
│   └── figures/                  # Figures
├── paper/
│   ├── AE5510_Problem1_Paper.tex # IEEE two-column report
│   ├── AE5510_Problem1_Paper.pdf
│   └── figures/
├── presentation/
│   └── AE5510_Problem1_Beamer.tex
├── scripts/
│   ├── run_attack.py
│   ├── verify_containment.py
│   ├── evaluate_transfer.py
│   ├── evaluate_binary_vs_continuous.py
│   └── compile_paper.py
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

---

## Getting started

### Install

```bash
git clone https://github.com/leeIITM/AE5510-Adversarial-Camouflage.git
cd AE5510-Adversarial-Camouflage
pip install -r requirements.txt
pip install -e .
```

### Run

```bash
# Optimise a camouflage on a sample scene (images go to outputs/adversarial_images/)
python scripts/run_attack.py

# Check that the sticker stays inside the object
python scripts/verify_containment.py

# Transfer to YOLO11m and YOLOv8s
python scripts/evaluate_transfer.py

# Two-colour vs. full-colour comparison
python scripts/evaluate_binary_vs_continuous.py

# Build the paper PDF
python scripts/compile_paper.py

# Run tests
pytest tests/ -v
```

---

## Limitations

- Everything was tested on digital images. No real stickers were printed or flown.
- The sticker is treated as flat, so curved or raised parts of an object (like an aircraft tail) are not modelled.
- Very few trucks and tanks were detected on clean images, so results for those classes are not reliable.

---

## Paper and citation

The full report is in [`paper/AE5510_Problem1_Paper.pdf`](paper/AE5510_Problem1_Paper.pdf).

```bibtex
@techreport{lee2026adversarial,
  title={Physically Grounded Binary Adversarial Camouflage for Real-Time Aerial Object Detectors Under Flight Dynamics},
  author={leeIITM},
  institution={Department of Aerospace Engineering, IIT Madras},
  type={AE5510 Machine Learning Applications Technical Report},
  year={2026},
  url={https://github.com/leeIITM/AE5510-Adversarial-Camouflage}
}
```

---

## License

MIT License. See the [LICENSE](LICENSE) file.
