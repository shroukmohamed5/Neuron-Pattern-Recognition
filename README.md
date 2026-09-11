# Single-Neuron Pattern Recognition with Hebb's Rule

A from-scratch implementation of a single artificial neuron trained with **Hebb's learning rule** to classify two custom 5×5 pixel-art letter patterns. Built with NumPy and Matplotlib in a Jupyter Notebook.

## Overview

This project explores how a single neuron — the simplest possible unit of a neural network — can learn to separate two classes of binary images using unsupervised Hebbian learning. It covers both classic **binary** and **bipolar** signal representations, threshold correction, and a custom activation function built around a confidence/uncertainty zone.

## What's inside

- **Pattern design** — two 5×5 pixel-art classes (2 reference images per class), with a controlled pixel-level difference within each class.
- **Hebbian training** — implementation of `w += t * x`, `b += t` from scratch, trained on both binary (0/1) and bipolar (-1/1) input encodings.
- **Decision boundary exploration** — a pixel-by-pixel morph between one image and the other, tracking the neuron's net input and output at every step to locate the exact decision boundary.
- **Similarity metrics** — Hamming distance and cosine similarity computed between every intermediate image and both class references.
- **Threshold correction** — automatic recalibration of the activation threshold based on the mean net input of each class.
- **Custom activation function** — a sigmoid-based activation configured from two data-driven cut points (Q1, Q2), splitting the net-input range into confident-class-A / uncertain / confident-class-B zones.
- **Visualizations** — image grids, transition tables, and an activation function plot with the reference points overlaid.

## Tech stack

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook

## Structure

The notebook (`lab1_neural_network.ipynb`) is organized into two main parts:

1. **Part 1 — Hebb's Rule**: pattern design, training (binary & bipolar), boundary detection, similarity analysis, threshold tuning.
2. **Part 2 — Custom Activation Function**: Q1/Q2 computation, sigmoid-based activation, uncertainty-zone detection along the transition path.

## Running it

```bash
pip install numpy matplotlib jupyter
jupyter notebook lab1_neural_network.ipynb
```

Run all cells top to bottom; each cell is self-contained and prints or plots its own output.

## Key takeaway

A single neuron can perfectly separate two linearly separable pattern classes, and the choice of signal encoding (binary vs. bipolar) meaningfully affects the strength of that separation — bipolar encoding, which treats "off" pixels as active negative signals rather than neutral zeros, produces a much wider margin between classes.
