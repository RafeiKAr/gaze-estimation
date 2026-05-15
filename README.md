
# Data-Efficient Personalization for Webcam-Based Eye Tracking

Master thesis project on gaze estimation and data-efficient personalization
using pretrained computer vision models.

---

## Project Overview

This project investigates webcam-based eye tracking using pretrained
visual encoders and lightweight personalization strategies.

The main objectives are:

- Build a complete gaze estimation pipeline
- Evaluate baseline methods
- Train regression models for 2D gaze prediction
- Investigate data-efficient personalization
- Compare sampling strategies:
  - Random sampling
  - Uncertainty-based sampling
  - Diversity-based sampling
- Evaluate generalization under domain shift
- Develop an interactive demo

---

## Baseline Experiments

Implemented baselines:

- Constant baseline
- Random baseline

Evaluation metrics:

- Euclidean distance
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Dataset

This project uses the:

GazeCapture Dataset

Official website:

https://gazecapture.csail.mit.edu/

Paper:

https://arxiv.org/abs/1609.06038

### Dataset Setup

Download the dataset manually and place it into:

data/raw/

Expected structure:

data/
└── raw/
    └── GazeCapture/
        ├── 00001/
        ├── 00002/
        └── ...

The dataset is NOT included in this repository.

---

## Installation

Create environment:

# bash
python -m venv venv

# Activate environment:
1) Linux / macOS
source venv/bin/activate

2) Windows
venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Project Structure
gaze-estimation/
│
├── data/
├── notebooks/
├── src/
├── results/
├── configs/
├── requirements.txt
├── README.md
└── main.py

# Running Baseline Evaluation
python main.py
or open:
notebooks/02_baseline.ipynb

# Example Results
| Method            | Mean Error |
| ----------------- | ---------- |
| Constant Baseline | 178 px     |
| Random Baseline   | 280 px     |


# Technologies
Python
PyTorch
OpenCV
NumPy
Matplotlib
Gradio

# Future Work
Pretrained visual backbones
Personalization strategies
Active learning
Domain adaptation
Online webcam demo

# Author
Master Thesis Project

# License
