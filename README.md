# Deep Learning-Based Multivariate Sensor Anomaly Detection

## Overview

This project detects abnormal behavior in multivariate sensor data using Dense Autoencoders and LSTM Autoencoders.

The system analyzes four sensor parameters:

* Temperature
* Pressure
* Vibration
* Humidity

It learns normal sensor behavior using reconstruction error and identifies unusual readings or time-series patterns.

## Methodology

Sensor Data → Preprocessing → Autoencoder / LSTM Autoencoder → Reconstruction Error → Threshold → Anomaly Detection

## Features

* Sensor data preprocessing and normalization
* Exploratory data analysis
* Dense Autoencoder for point anomaly detection
* LSTM Autoencoder for temporal anomaly detection
* Threshold-based anomaly classification
* Normal / Warning / Critical severity levels
* Interactive sensor monitoring
* Model evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix

## Results

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  95.30% |
| Precision |  51.55% |
| Recall    | 100.00% |
| F1-Score  |  68.03% |

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Google Colab

## How to Run

1. Open `dl.ipynb` in Google Colab or Jupyter Notebook.
2. Run all cells sequentially.
3. Review the generated graphs and evaluation metrics.
4. Test the interactive sensor anomaly detector with custom sensor values.

## Project Purpose

Developed as an academic AI/ML major project to demonstrate deep learning-based multivariate sensor anomaly detection and monitoring.
