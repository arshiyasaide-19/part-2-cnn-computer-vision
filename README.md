# Part 2 — CNN Computer Vision: Product Defect Detection

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** to classify product images into four defect categories:
- `normal` — no defect
- `scratch` — surface scratch
- `dent` — dent/indentation
- `stain` — surface stain

The model is trained on a synthetic dataset of **400 images (100 per class)** generated programmatically using NumPy and PIL.

---

## 📁 Repository Structure

```
part-2-cnn-computer-visionn/
│
├── notebook.ipynb              ← Main Jupyter/Colab notebook (all 11 cells)
├── labels.csv                  ← Image filename ↔ class label mapping
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── results/
│   ├── training_curves.png         ← Accuracy & Loss plots
│   ├── confusion_matrix.png        ← Confusion matrix heatmap
│   ├── model_comparison_table.csv  ← Hyperparameter experiment results
│
└── sample_predictions/
    └── sample_predictions.png      ← Grid of 8 test predictions
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebook.ipynb`
3. Run all cells top to bottom (`Runtime → Run all`)
4. No dataset upload needed — Cell 2 auto-generates all images!

### Option 2 — Local (Jupyter Notebook)
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## 🧠 Model Architecture

| Layer | Type | Output Shape | Details |
|-------|------|--------------|---------|
| 1 | Conv2D | (62,62,32) | 32 filters, 3×3, ReLU |
| 2 | MaxPooling2D | (31,31,32) | 2×2 |
| 3 | Conv2D | (29,29,64) | 64 filters, 3×3, ReLU |
| 4 | MaxPooling2D | (14,14,64) | 2×2 |
| 5 | Conv2D | (12,12,128) | 128 filters, 3×3, ReLU |
| 6 | MaxPooling2D | (6,6,128) | 2×2 |
| 7 | Flatten | (4608,) | — |
| 8 | Dense | (128,) | ReLU |
| 9 | Dropout | (128,) | 0.3 rate |
| 10 | Dense (output) | (4,) | Softmax |

**Optimizer:** Adam | **Loss:** Categorical Crossentropy | **Epochs:** 20

---

## 📊 Hyperparameter Experiments

| Experiment | Filters | Dropout | Batch Size | Epochs | Test Accuracy |
|------------|---------|---------|------------|--------|---------------|
| Exp 1 — Baseline | 16 | 0.2 | 32 | 15 | ~72% |
| Exp 2 — More Filters | 32 | 0.3 | 32 | 20 | ~85% |
| Exp 3 — High Dropout | 32 | 0.5 | 16 | 20 | ~78% |

> **Best model:** Experiment 2 — 32 filters, 0.3 dropout, 20 epochs.

---

## 🔍 Key Findings

- **Convolutional layers** detect spatial patterns like edges, textures, and shapes at different scales.
- **MaxPooling** reduces spatial dimensions, lowers computation, and improves translation invariance.
- **Dropout (0.3)** was the sweet spot — high enough to prevent overfitting, low enough to not underfit.
- Increasing filters from 16 → 32 had the **largest positive impact** on accuracy.
- The model converged by epoch 15 with no signs of significant overfitting.

---

## 📦 Dataset

Synthetically generated — **no external dataset required**.

| Class | Images | Description |
|-------|--------|-------------|
| normal | 100 | Uniform light gray surface |
| scratch | 100 | Gray surface with horizontal dark lines |
| dent | 100 | Gray surface with circular dark patch |
| stain | 100 | Gray surface with brownish circular blob |

---

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow / Keras 2.x
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- Pillow (PIL)

---

## 👩‍💻 Author

**Arshiya Saide** | CA Final Student & Business Analytics Learner  
Nagpur, Maharashtra, India
