# Convolutional Neural Network (CNN) for Image Classification

This repository contains the source code and experimental results for **Experiment 3** of the Deep Learning Laboratory course. The experiment focuses on designing and implementing Convolutional Neural Networks (CNNs) for image classification tasks using the TensorFlow/Keras deep learning framework.

The experiment uses the **CIFAR-10 image dataset** to study CNN building blocks including convolution layers, activation functions, pooling operations, flattening, and fully connected layers. It also explores CNN hyperparameters, feature extraction, model evaluation, and visualization of learned representations.

---

## Contents

### 1. **CNN Image Classification using CIFAR-10** (`cnn.ipynb`)

The notebook implements a complete CNN-based image classification pipeline.

Features include:

* Loading and preprocessing the CIFAR-10 dataset.
* Normalizing image pixel values.
* Performing one-hot encoding for multi-class classification.
* Designing a CNN architecture using TensorFlow/Keras.
* Implementing:
  * Convolutional layers
  * ReLU activation functions
  * Pooling layers
  * Flattening layers
  * Fully connected layers
  * Dropout regularization
* Training the CNN model using the Adam optimizer.
* Evaluating model performance using:
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
  * Classification Report

---

## 2. **CNN Architecture and Hyperparameter Analysis**

The experiment analyzes the effect of CNN design choices on model performance.

The following hyperparameters are studied:

* Kernel size
* Number of convolution filters
* Stride
* Padding
* Optimizer selection
* Batch size
* Number of epochs

The notebook also includes:

* Model summary and parameter analysis.
* Trainable parameter count of convolution layers.
* Comparison of different pooling techniques:
  * MaxPooling
  * AveragePooling

---

## 3. **Feature Map Visualization**

Intermediate convolution outputs are extracted and visualized to understand CNN feature learning.

The visualization demonstrates how CNN layers progressively learn:

* Edges
* Textures
* Shapes
* Object patterns

Generated feature maps are stored in the `images/` directory.

---

## 4. **Training and Evaluation Visualizations**

The `images/` folder contains all generated experimental results.

Contents:

```
images/
│
├── sample_images.png
│   └── Sample CIFAR-10 images with labels
│
├── class_distribution.png
│   └── Distribution of CIFAR-10 classes
│
├── training_accuracy.png
│   └── Training and validation accuracy curves
│
├── training_loss.png
│   └── Training and validation loss curves
│
├── confusion_matrix.png
│   └── Class-wise prediction performance
│
├── feature_maps.png
│   └── CNN convolution feature visualizations
│
├── pooling_comparison.png
│   └── MaxPooling vs AveragePooling comparison
│
└── cnn_architecture.png
    └── CNN model architecture diagram
```

---

## 5. **Report** (`report.pdf`)

- soon

---

# Project Structure

```
Deep-Learning-Lab/
│
├── cnn.ipynb
├── report.pdf
├── requirements.txt
└── images/
```

---

# Installation and Setup

## 1. Clone Repository

```bash
git clone https://github.com/sadhumitha-s/Deep-Learning-Lab.git
cd 3-cnn
```

---

## 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate
```
---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
cnn.ipynb
```

Run all cells to train the CNN model and generate outputs.

---

# Dataset

The experiment uses the CIFAR-10 dataset.

Dataset information:

| Parameter | Value |
|---|---|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 × 32 × 3 |
| Number of Classes | 10 |

Classes:

```
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
```

The dataset is automatically downloaded using:

```python
tensorflow.keras.datasets.cifar10
```

---

# CNN Architecture

Implemented CNN structure:

```mermaid
graph TD
    A["Input Image (32x32x3)"] --> B["Conv2D (32 filters, 3x3 kernel, ReLU)"]
    B --> C["MaxPooling2D"]
    C --> D["Conv2D (64 filters, 3x3 kernel, ReLU)"]
    D --> E["MaxPooling2D"]
    E --> F["Conv2D (128 filters, 3x3 kernel, ReLU)"]
    F --> G["Flatten"]
    G --> H["Dense Layer (ReLU)"]
    H --> I["Dropout"]
    I --> J["Dense Output (Softmax, 10 classes)"]
    
    classDef layer fill:#f9f9f9,stroke:#333,stroke-width:1px;
    class A,B,C,D,E,F,G,H,I,J layer;
```

---

# Evaluation Metrics

The CNN model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

---

# Technologies Used

* Python 3.8+
* TensorFlow
* Keras
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---
