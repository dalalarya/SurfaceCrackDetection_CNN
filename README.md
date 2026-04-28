#  Industrial Surface Crack Detection using CNN

A deep learning project that detects surface cracks in industrial images using 
a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

Surface cracks in industrial structures can lead to serious safety hazards if 
undetected. Manual inspection is slow and error-prone. This project automates 
crack detection using computer vision and deep learning.

##  Dataset

- ~40,000 images total
- 2 Classes: `Crack` and `NoCrack`
- Split: 70% Training | 15% Validation | 15% Testing

---

##  Model Architecture

| Layer | Details |
|---|---|
| Conv2D (x4) | Filters: 32 → 64 → 128 → 256, ReLU activation |
| BatchNormalization | After each Conv layer |
| MaxPooling2D | After each Conv+BN block |
| Dense (256) | ReLU + Dropout 0.5 |
| Dense (128) | ReLU + Dropout 0.3 |
| Output (1) | Sigmoid (Binary Classification) |

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- Scikit-learn (evaluation)

---

## How to Run

### 1. Clone the repository
git clone https://github.com/dalalarya/Surface-Crack-Detection-CNN.git
cd Surface-Crack-Detection-CNN

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your dataset
Place your dataset in the following structure:
CrackDataset/
├── Positive/   ← Crack images
└── Negative/   ← NoCrack images

### 4. Run the model
python crack_detection.py

---

## 📈 Training Strategy

- **Data Augmentation:** Rotation, zoom, horizontal flip (training set only)
- **EarlyStopping:** Stops training when val_loss stops improving (patience=4)
- **ModelCheckpoint:** Saves best model based on val_accuracy
- **ReduceLROnPlateau:** Reduces learning rate when progress stalls

---

## 📉 Evaluation

Model evaluated using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Real-time single image prediction function

---

## Key Learnings

- Overfitting prevention using BatchNormalization and Dropout
- Importance of train/val/test split for unbiased evaluation
- Effect of data augmentation on model generalization
- Using callbacks for automated model optimization

---

##  Author

**Arya Anand Dalal**  
B.Tech in AI & Data Science — Vishwakarma University, Pune  
[LinkedIn](https://www.linkedin.com/in/arya-dalal-336b9127a) | 
