# DCARS: Trajectory-Level Detection of Compositional Safety Failures in Multi-Turn AI Systems

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## Overview

This repository contains the official implementation of:

**DCARS: Dynamic Context-Aware Risk Scoring**

A trajectory-aware safety evaluation framework for detecting **compositional safety failures** in multi-turn Large Language Model (LLM) interactions.

Unlike traditional static safety evaluation approaches that analyze responses independently, DCARS models **cumulative conversational risk** across interaction trajectories.

---

## Key Features

* Trajectory-level safety evaluation
* Dynamic conversational risk accumulation
* Compositional failure detection
* DistilBERT-based probabilistic risk modeling
* Multi-turn interaction trajectory construction
* Temporal safety modeling for LLM systems
* Comparative static vs dynamic evaluation

---

## Framework Architecture

```text
PKU-SafeRLHF Dataset
          │
          ▼
Train Risk Classifier (DistilBERT)
          │
          ▼
Response-Level Risk Scoring
          │
          ▼
HH-RLHF Multi-turn Sequences
          │
          ▼
Dynamic Risk Accumulation (DCARS)
          │
          ▼
Trajectory-Level Safety Detection
```

---

## Repository Structure

```bash
DCARS/
│
├── data/
│   ├── processed/
│   ├── trajectories/
│   └── risk_scores/
│
├── models/
│   ├── risk_classifier.py
│   ├── dcars.py
│   └── baseline.py
│
├── scripts/
│   ├── train_classifier.py
│   ├── generate_trajectories.py
│   ├── evaluate_dcars.py
│   └── visualize_results.py
│
├── notebooks/
│   ├── analysis.ipynb
│   └── visualization.ipynb
│
├── results/
│   ├── plots/
│   ├── metrics/
│   └── checkpoints/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ShakibulAkash/DCARS-Trajectory-Level-Detection-of-Compositional-Safety-Failures-in-Multi-Turn-AI-Systems.git

cd DCARS-Trajectory-Level-Detection-of-Compositional-Safety-Failures-in-Multi-Turn-AI-Systems
```

---

## Create Environment

```bash
python -m venv dcars_env
```

### Linux / MacOS

```bash
source dcars_env/bin/activate
```

### Windows

```bash
dcars_env\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```txt
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
accelerate
```

---

# Dataset Setup

## PKU-SafeRLHF

Used for training the response-level risk classifier.

```python
from datasets import load_dataset

pku_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
```

---

## HH-RLHF

Used for constructing multi-turn conversational trajectories.

```python
from datasets import load_dataset

hh_dataset = load_dataset("Anthropic/hh-rlhf")
```

---

# Training the Risk Model

Train the DistilBERT-based probabilistic risk classifier:

```bash
python scripts/train_classifier.py
```

---

## Risk Function

DCARS models response-level risk as:

[
f(o_t) = P(unsafe \mid o_t)
]

The classifier outputs continuous probabilistic safety scores.

---

# Multi-Turn Trajectory Construction

Generate controlled interaction trajectories:

```bash
python scripts/generate_trajectories.py
```

---

## Trajectory Patterns

### Safe Trajectories

```text
Low Risk → Mid Risk → Mid Risk
```

### Unsafe Trajectories

```text
Low Risk → Mid Risk → High Risk
```

---

# DCARS Dynamic Risk Accumulation

Run trajectory-aware safety evaluation:

```bash
python scripts/evaluate_dcars.py
```

---

## DCARS Equation

The accumulated conversational risk is recursively computed as:

[
R_t = \alpha R_{t-1} + (1 - \alpha)f(o_t)
]

Where:

* (R_t) = accumulated conversational risk
* (f(o_t)) = response-level risk probability
* (\alpha) = temporal accumulation parameter

---

# Baseline Static Evaluation

Run standard response-level safety evaluation:

```bash
python models/baseline.py
```

---

# Visualization

Generate trajectory analysis plots:

```bash
python scripts/visualize_results.py
```

---

# Experimental Results

| Method            | F1 Score | AUC    |
| ----------------- | -------- | ------ |
| Static Evaluation | 0.230    | —      |
| DCARS             | 0.473    | 0.9094 |

---

# Compositional Failure Detection

| Metric                | Value |
| --------------------- | ----- |
| Total CF Cases        | 37    |
| Static Detection Rate | 0.0   |
| DCARS Detection Rate  | 1.0   |

---

# Example Usage

```python
from models.dcars import DCARS

model = DCARS(alpha=0.5)

risk_scores = [0.15, 0.28, 0.41]

trajectory_risk = model.compute(risk_scores)

print(trajectory_risk)
```

---

# Main Contributions

* Introduces trajectory-level safety evaluation
* Detects compositional conversational failures
* Models cumulative conversational risk dynamically
* Demonstrates limitations of static safety evaluation
* Establishes temporal risk modeling for future AI safety systems

---

# Citation

```bibtex
@inproceedings{akash2026dcars,
  title={DCARS: Trajectory-Level Detection of Compositional Safety Failures in Multi-Turn AI Systems},
  author={Akash, Shakibul Islam and Basfoor, Rima and Salam, Abdus},
  booktitle={Proceedings of the International Conference on Computing Advancements (ICCA)},
  year={2026}
}
```

---

# Paper

DCARS introduces a trajectory-aware framework for cumulative conversational risk modeling in multi-turn AI systems. The framework dynamically accumulates risk across conversational steps to detect compositional safety failures that remain invisible under traditional response-level evaluation methods.

---

# License

This project is released under the MIT License.

---

# Authors

* Shakibul Islam Akash
* Rima Basfoor
* Abdus Salam

---

# Acknowledgement

This work was conducted with support from American International University–Bangladesh (AIUB).
