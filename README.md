# Differential-Testing

## Overview
This project performs differential testing on at least two ResNet50 models trained on CIFAR-10.
It detects disagreement-inducing inputs, measures neuron coverage, and saves suspicious examples.

## Models

- Architecture: ResNet50
- Dataset: CIFAR-10
- Two models trained with different random seeds

## DeepXplore Usage

We analyzed the original DeepXplore repository and adopted its core ideas:
- Differential testing across multiple DNNs
- Neuron coverage maximization

Due to compatibility issues (TensorFlow 1.x), we reimplemented the framework in PyTorch using ResNet50 on CIFAR-10.

## Environment

- Recommended Python version: **Python 3.10**

```bash
pip install -r requirements.txt
python.exe -m pip install --upgrade pip
```

This project was tested with the following GPU environment:

CUDA: 11.8
cuDNN: 8.1

## Modifications for CIFAR-10

The original DeepXplore implementation was designed for the MNIST dataset using three simple convolutional models and TensorFlow 1.x APIs.

To adapt it for CIFAR-10 and ResNet-based models, the following modifications were applied:

### 1. Dataset Replacement

- Replaced **MNIST (28×28 grayscale)** with **CIFAR-10 (32×32 RGB)**
- Updated preprocessing:
  - Normalized pixel values to [0, 1]
  - Added resizing step to **224×224** for compatibility with ResNet50

### 2. Model Replacement

- Removed original models (`Model1`, `Model2`, `Model3`)
- Replaced with:
  - **Two pretrained ResNet50-based models**
- Models are loaded using:
  ```python
  load_model("model1.h5")
  load_model("model2.h5")