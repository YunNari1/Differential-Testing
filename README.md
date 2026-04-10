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
```

## GPU (Optional)

This project runs on CPU by default.

If you want to use GPU, install PyTorch with CUDA (Python 3.10 or 3.11 recommended):

```bash
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```