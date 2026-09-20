# Cherry Leaf Disease Classification Pipeline

An end-to-end Deep Learning classification system built to detect and categorize diseases in cherry leaves using Transfer Learning (`MobileNetV2`), optimized with Mixed Precision and distributed multi-GPU strategies in TensorFlow/Keras.

---

## 📌 Project Overview

Plant disease identification is crucial for precision agriculture. This repository contains a deep learning pipeline designed to classify cherry leaf images into five distinct health/disease categories:
1. **Cherry Normal leaf**
2. **Cherry Leaf Scorch**
3. **Cherry brown_spot**
4. **Cherry purple leaf spot**
5. **Cherry_shot hole disease**

---

## 🔍 Exploratory Data Analysis (EDA) & Dataset Structure

- **Total Dataset Size**: 3,338 images across 5 classes.
- **Class Distribution & Imbalance**:
  - *Majority Class*: Cherry Leaf Scorch (~1,097 images)
  - *Minority Class*: Cherry_shot hole disease (~264 images)
- **Metadata Check**: 
  - All sampled images maintain a uniform resolution of `800 x 1000` pixels with a consistent aspect ratio of `0.8`. 
  - Images are globally resized to fit standard deep learning architectures (e.g., `224x224`).

---

## ⚙️ Performance Optimizations & Infrastructure

To maximize training efficiency and leverage hardware capabilities, the pipeline implements:
- **Mixed Precision Training**: Enabled `mixed_float16` compute policy to accelerate training speed and reduce memory footprint on modern GPUs.
- **Distributed Strategy**: Utilizes TensorFlow's `MirroredStrategy` for scalable multi-GPU execution.
- **Reproducibility**: Global random seeding (`seed=42`) across NumPy, Python Random, and TensorFlow operations.

---

## 🛠️ Tech Stack & Frameworks

- **Language**: Python
- **Deep Learning**: TensorFlow, Keras, MobileNetV2
- **Data Handling & Visualization**: Pandas, NumPy, Matplotlib, Seaborn, PIL
- **Environment**: Google Colab (GPU-accelerated)
