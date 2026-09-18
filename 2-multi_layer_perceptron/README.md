# Multi-Layer Perceptron (MLP) for Multi-Class Classification

This repository contains the source code and experimental results for Experiment 2 of the Deep Learning Laboratory (CS3807) course. It includes implementations for a Multi-Layer Perceptron (MLP) on the Fashion-MNIST dataset and an implementation of a Multi-Layer Perceptron from scratch for the XOR logic gate.

## Contents

1. **Multi-Layer Perceptron (MLP) for Multi-Class Classification** (`mlp.ipynb`)

   * Preprocesses the Fashion-MNIST image dataset (flattening, normalizing, and one-hot encoding).
   * Constructs and trains a baseline MLP using TensorFlow/Keras.
   * Evaluates the model using Accuracy, Precision, Recall, and F1-score.
   * Performs automated hyperparameter optimization using `RandomizedSearchCV` and `SciKeras`.
   * Generates confusion matrices and loss/accuracy plots.

2. **Multi-Layer Perceptron for XOR Gate** (`xor.ipynb`)

   * Implements a two-layer neural network from scratch using NumPy.
   * Uses sigmoid activation functions and the backpropagation algorithm for training.
   * Learns the non-linearly separable XOR function successfully.
   * Visualizes the decision boundary at regular training intervals (Epochs 0, 100, 200, 300, 400, 500, 600, 700, 744, and the final model).
   * Displays the final learned weights after convergence.

3. **Report** (`report.pdf`)

   * The compiled report integrating the results, hyperparameters, plots, and analysis for both experiments.

4. **Images**

   * `images/`: Contains all plots generated from the Fashion-MNIST MLP training and hyperparameter search.
   * `images_xor/`: Contains decision boundary visualizations generated during the XOR MLP training process.

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sadhumitha-s/Deep-Learning-Lab.git
   cd 2-multi_layer_perceptron
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks:**

   ```bash
   jupyter notebook
   ```

   Open `mlp.ipynb` and `xor.ipynb` in your browser.

## Requirements

The project relies on standard data science and deep learning libraries, specifically tailored to use `SciKeras` for hyperparameter tuning and NumPy for implementing the XOR neural network from scratch. See `requirements.txt` for exact versions.

* Python 3.8+
* TensorFlow / Keras
* SciKeras
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
