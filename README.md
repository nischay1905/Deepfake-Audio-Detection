# Deepfake Audio Detection 

### MARS Club IIT Roorkee – Open Projects 2026

---

##  Project Overview

With the rapid advancement of Generative AI technologies, creating realistic synthetic speech has become easier than ever. While these technologies offer many benefits, they also introduce serious challenges related to misinformation, identity theft, voice cloning, and digital fraud.

This project presents a Deep Learning-based system capable of distinguishing between genuine human speech and AI-generated deepfake audio. The model converts speech recordings into Mel Spectrogram representations and leverages EfficientNetB0 to learn discriminative patterns between authentic and synthetic speech.

The proposed system demonstrates strong performance on unseen audio samples and successfully satisfies all project verification requirements.

---

##  Problem Statement

Develop a Machine Learning or Deep Learning system capable of classifying speech recordings as:

- Genuine (Human Speech)
- Deepfake (AI-Generated Speech)

The model should generalize well to previously unseen audio samples and demonstrate robust performance on the evaluation dataset.

---

### Live Demo

Add your Streamlit deployment link here:

https://your-streamlit-app-link.streamlit.app

##  Dataset

Dataset Used:

**The Fake-or-Real (FoR) Dataset**

The dataset contains thousands of audio recordings categorized into:

- Genuine Human Speech
- AI Generated Speech

Dataset Structure:

- Training Set
- Validation Set
- Testing Set

For this implementation, approximately **20,000 audio samples** were used for training and evaluation.

---

## ⚙️ Methodology

### 1. Data Collection

Audio samples were loaded from the Fake-or-Real dataset and organized into training, validation, and testing sets.

### 2. Audio Preprocessing

Each audio file was:

- Loaded using Librosa
- Converted to mono audio
- Resampled to a uniform sampling rate
- Converted into Mel Spectrograms

### 3. Feature Extraction

Mel Spectrograms were generated for each audio sample.

Why Mel Spectrograms?

- Capture frequency information effectively
- Widely used in speech recognition
- Preserve important speech characteristics
- Work well with CNN-based architectures

### 4. Data Preparation

- Spectrogram Size: 128 × 128
- Single channel converted to RGB format
- Dataset shuffled before training
- Stratified train-test split applied

### 5. Model Architecture

Transfer Learning was used through EfficientNetB0.

Architecture:

Input Spectrogram
↓
EfficientNetB0
↓
Global Average Pooling
↓
Dense Layer (128)
↓
Dropout (0.4)
↓
Sigmoid Output Layer

---

## Deep Learning Model

Model Used:

- EfficientNetB0
- TensorFlow / Keras

Training Configuration:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 64
- Epochs: 15
- Early Stopping
- Learning Rate Scheduling

---

## Performance Results

### Overall Metrics

| Metric | Score |
|----------|----------|
| Accuracy | 95.45% |
| F1 Score | 95.50% |
| Equal Error Rate (EER) | 4.47% |

### Verification Requirements

| Requirement | Threshold | Achieved |
|-------------|------------|-----------|
| Accuracy | ≥ 80% | ✅ 95.45% |
| F1 Score | ≥ 80% | ✅ 95.50% |
| EER | ≤ 12% | ✅ 4.47% |
| Per-Class Accuracy | ≥ 75% | ✅ Passed |

---

##  Confusion Matrix

|               | Predicted Genuine | Predicted Deepfake |
|---------------|-------------------|--------------------|
| Actual Genuine | 1887 | 123 |
| Actual Deepfake | 59 | 1931 |

---



---
## 📁 Repository Structure

```text
Deepfake-Audio-Detection
│
├── notebook
│   └── Deepfake_Audio_Detection.ipynb
│
├── models
│   └── deepfake_audio_model_20k.keras
│
├── reports
│   └── confusion_matrix.png
│
├── app.py
│
├── requirements.txt
│
└── README.md
```












MARS Club IIT Roorkee – Open Projects 2026

---

## 📜 License

This project is developed for educational and research purposes under the MARS Club IIT Roorkee Open Projects Program.
