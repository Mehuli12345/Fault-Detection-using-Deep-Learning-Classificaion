# 🎯 Fault Detection from Audio Signals using LSTM (MATLAB)

This project addresses the task of detecting mechanical faults in machinery using **audio signal analysis** with **deep learning**. The solution is implemented entirely in **MATLAB** and employs a **BiLSTM (Bidirectional Long Short-Term Memory)** neural network to classify different types of faults based on their acoustic signatures.

Mechanical systems often emit distinct sounds when operating under faulty conditions. Capturing and analyzing these acoustic patterns enables early fault detection, which is critical for predictive maintenance, safety, and system reliability.

---

## 📌 Objective

To build an end-to-end deep learning pipeline that can:
- Preprocess audio recordings of machinery (e.g., compressors)
- Extract meaningful features using MFCCs and their temporal derivatives
- Train a deep neural network (BiLSTM) to classify different fault types
- Evaluate model performance using multiple statistical metrics
- Visualize the results using plots such as the **confusion matrix** and **ROC curve**

---

## 🔍 Key Features

### 1. Audio-Based Fault Classification
Each sample in the dataset is a `.wav` audio recording of a machine under a specific condition — either healthy or faulty. The goal is to classify these conditions directly from audio.

### 2. Feature Extraction with MFCC + Deltas
- **MFCCs (Mel-Frequency Cepstral Coefficients)** capture spectral information of the sound.
- First- and second-order temporal derivatives (Δ and ΔΔ) are also extracted to represent dynamic aspects of the signal.
- The result is a 39-dimensional feature vector per frame (13 MFCC + 13 Δ + 13 ΔΔ), forming a time series input for the model.

### 3. Dataset Balancing
To handle class imbalance, minority class samples are **augmented using Gaussian noise** (jittering), which enhances generalization without collecting more data.

### 4. Deep Learning Architecture
- Input: Time series of MFCC-based features
- Layers:
  - 1D Convolutional Layer for local pattern learning
  - Batch Normalization for stability
  - Bidirectional LSTM layers to capture forward and backward temporal dependencies
  - Dropout layers for regularization
  - Fully connected output with Softmax activation for classification

### 5. Training Strategy
- **5-fold Cross-Validation** is used to ensure model generalization and robustness.
- The model is trained using the Adam optimizer with a decaying learning rate.

---

## 📊 Evaluation Metrics

The trained model is evaluated using the following metrics:
- **Accuracy**: Overall classification correctness
- **Precision, Recall, F1-score**: Per-class and macro-averaged
- **Matthews Correlation Coefficient (MCC)**: Measures the quality of binary and multiclass classifications, even with imbalanced datasets
- **Confusion Matrix**: Visual display of true vs predicted classes
- **ROC Curve**: Multi-class ROC curves and AUC (Area Under Curve) values

---

## 🧪 Results Summary

The model achieves strong classification performance, often exceeding **95% accuracy** with high F1 and MCC scores. This indicates reliable fault discrimination across different categories.

---

## 🧠 Applications

- Predictive maintenance in manufacturing
- Real-time monitoring of industrial machinery
- Fault diagnosis in HVAC systems, compressors, motors, etc.
- Smart factories and Industry 4.0 automation

---

## 📁 Project Structure

- `faultDetection_mainCode.mlx`: Complete end-to-end implementation (single file)
- Function sections within the code:
  - `extractFeaturesFromAudio`: Extracts MFCCs and deltas from `.wav` files
  - `balanceDataset`: Augments dataset with jittered signals
  - `buildLSTM`: Defines the BiLSTM-based model architecture
  - `evaluateModel`: Calculates metrics and returns predictions
  - `plotROC`: Plots multi-class ROC curves using one-hot encoded labels

---

## 📘 Summary

This MATLAB project showcases how **deep learning can effectively classify machine conditions using audio signals**, even with limited data. The model generalizes well through feature-rich inputs (MFCCs), careful augmentation, and a deep recurrent architecture — making it highly suitable for real-world industrial fault diagnosis tasks.
