# 🌾 Rice Classification with PyTorch

> A binary classification project using a custom neural network built from scratch with PyTorch to classify rice grains by their morphological features.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivanshu1512/Rice_Classification_PyTorch/blob/main/Tabular_classification.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~98%25-brightgreen)

---

## 📌 Overview

This project demonstrates how to build a **tabular binary classifier** using PyTorch on the [Rice Type Classification dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) from Kaggle.

The goal is to classify rice grains as one of two types — **Jasmine (1)** or **Gonen (0)** — based on their physical and geometric measurements.

The notebook covers the complete ML pipeline:

- 📥 Data loading & cleaning
- 🔢 Normalization / preprocessing
- 🔗 Custom PyTorch `Dataset` and `DataLoader`
- 🧠 Neural network model definition with `nn.Module`
- 🔁 Training loop with real-time validation tracking
- 📊 Loss & accuracy visualization

---

## 📊 Dataset

| Property     | Details                                   |
|--------------|-------------------------------------------|
| Source       | Kaggle — mssmartypants                    |
| Rows         | 18,185                                    |
| Input Features | 10 numeric (geometric measurements)     |
| Target       | Binary — `0` = Gonen, `1` = Jasmine       |
| Class Balance | Roughly balanced (~50/50)               |

**Features used:** `Area`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Perimeter`, `Roundness`, `AspectRation`

---

## 🏗️ Model Architecture

A lightweight **fully-connected neural network** with a single hidden layer:

```
Input (10 features)  →  Linear(10, 10)  →  Linear(10, 1)  →  Sigmoid  →  Binary Output
```

| Layer    | In | Out | Params |
|----------|----|-----|--------|
| Linear-1 | 10 | 10  | 110    |
| Linear-2 | 10 | 1   | 11     |
| Sigmoid  | —  | —   | 0      |
| **Total**|    |     | **121**|

---

## ⚙️ Hyperparameters

| Parameter     | Value    |
|---------------|----------|
| Batch Size    | 32       |
| Epochs        | 10       |
| Hidden Neurons| 10       |
| Learning Rate | 1e-3     |
| Optimizer     | Adam     |
| Loss Function | BCELoss  |

---

## 📈 Training Results

| Metric              | Value    |
|---------------------|----------|
| Train Accuracy      | ~98.75%  |
| Validation Accuracy | ~98.24%  |
| Data Split          | 70 / 15 / 15 (train / val / test) |

The model converges quickly — loss drops sharply in the first 2 epochs and stabilises by epoch 5, showing that the features are well-separated between classes.

---

## 🚀 Getting Started

### Run in Google Colab (Recommended)

Click the **Open in Colab** badge at the top, or [click here](https://colab.research.google.com/github/shivanshu1512/Rice_Classification_PyTorch/blob/main/Tabular_classification.ipynb).

### Run Locally

```bash
git clone https://github.com/shivanshu1512/Rice_Classification_PyTorch.git
cd Rice_Classification_PyTorch

pip install torch torchsummary scikit-learn matplotlib pandas numpy opendatasets

jupyter notebook Tabular_classification.ipynb
```

> **Note:** You will need [Kaggle API credentials](https://www.kaggle.com/docs/api) to download the dataset when prompted.

---

## 🧠 PyTorch Concepts Covered

| Concept | Description |
|---------|-------------|
| `nn.Module` | Base class for all neural network models |
| `Dataset` & `DataLoader` | Custom data pipeline with batching |
| `nn.Linear` | Fully connected (dense) layers |
| `nn.Sigmoid` | Output activation for binary classification |
| `nn.BCELoss` | Binary Cross-Entropy loss function |
| `Adam` optimizer | Adaptive learning rate optimisation |
| `torch.no_grad()` | Disabling gradients during validation/inference |
| `.backward()` & `.step()` | Backpropagation and weight update |

---

## 👤 Author

**Shivanshu Shukla**
- 🔗 [GitHub](https://github.com/shivanshu1512)
- 📊 [Kaggle](https://www.kaggle.com/meshivanshu)

---

*Made with ❤️ and PyTorch*