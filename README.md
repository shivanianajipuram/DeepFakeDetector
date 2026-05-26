# Deepfake Detection System


## Project Description

This is a deep learning-based web application that detects whether an uploaded image is **REAL** or **FAKE (Deepfake)**.

The system uses Transfer Learning with **MobileNetV2** for efficient and accurate image classification.

The application is optimized for:

- Fast prediction
- Lightweight deployment
- Real-time image detection
- Streamlit web interface

---

# Dataset

The model is trained on a large-scale image dataset containing:

- Real human face images
- AI-generated / manipulated fake face images
  
### Download dataset

```bash
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
```
- unzip the file
- copy+paste the path in train.py
Dataset structure:

```bash
Dataset/
│
├── train/
│   ├── Real/
│   └── Fake/
│
└── test/
    ├── Real/
    └── Fake/
```

---

# Preprocessing Pipeline

## Image Loading

- Images loaded using TensorFlow/Keras
- RGB image format
- Images resized to:

```bash
160 x 160
```

---

## Data Augmentation

Applied to improve generalization and reduce overfitting:

- Rotation
- Zoom
- Horizontal Flip
- Width Shift
- Height Shift
- Brightness Adjustment

---

## Normalization

Pixel values scaled using:

```bash
rescale = 1./255
```

---

# Model Architecture

The project uses **MobileNetV2 Transfer Learning**.

## Base Model

```bash
MobileNetV2(
    include_top=False,
    weights='imagenet'
)
```

Pretrained on millions of images from ImageNet.

---

## Classification Head

```bash
GlobalAveragePooling2D
↓
Dropout(0.5)
↓
Dense(1, activation='sigmoid')
```

---

# Overfitting Prevention

The following techniques were used:

- Transfer Learning
- Dropout Layer
- Data Augmentation
- Validation Testing
- Large Dataset Training

---

# Streamlit Deployment

The application is deployed using Streamlit with a clean and simple UI.

## Features

- Upload image directly
- Detect REAL or FAKE image
- Fast prediction
- Minimal UI
- Lightweight deployment

---

# File Structure

```bash
Project/
│
├── app.py
├── train.py
├── best_deepfake_model.h5
├── requirements.txt
├── .gitignore
└── Dataset/
    ├── train/
    │   ├── Real/
    │   └── Fake/
    │
    └── test/
        ├── Real/
        └── Fake/
```

---

# Model Training Pipeline

## Image Generator

```bash
ImageDataGenerator
```

Used for:

- preprocessing
- augmentation
- batching

---

## Transfer Learning Workflow

1. Load pretrained MobileNetV2
2. Freeze base layers
3. Add custom classification head
4. Train final layers
5. Save trained model

---

# Streamlit Application Workflow

## User Uploads Image

```bash
Upload Image
```

↓

## Image Preprocessing

```bash
Resize → Normalize → Expand Dimensions
```

↓

## Prediction

```bash
REAL IMAGE
or
FAKE IMAGE
```

---

# Usage Instructions

## 1. Clone Repository Locally

```bash
git clone https://github.com/shivanianajipuram/DeepFakeDetector.git

cd deepfake-detector
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Train Model

```bash
python train.py
```

---
## 4. Convert model .keras to .h5
```bash
python convert_model.py
```

## 5. Run Streamlit App

```bash
streamlit run app.py
```

---

# Requirements

```bash
Python >= 3.10 (prefer-3.10.13)
tensorflow
keras
numpy
pillow
streamlit
scikit-learn
matplotlib
opencv-python
```

---

# Deployment Notes

For deployment platforms like Render:

## Environment Variable

```bash
PYTHON_VERSION = 3.10.13
```
## Live Demo
```bash
https://deepfakedetector-1.onrender.com/
```
