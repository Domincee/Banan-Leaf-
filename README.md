# 🍌 Banana Leaf Disease Detector

A machine learning web application built with **Python, OpenCV, scikit-learn, and Flask** to detect whether a banana leaf is **Healthy**, **Unhealthy**, or **Not a Leaf** using image processing and KNN classification.

---

## 📖 Project Overview

This project extracts texture and color-based features (using **GLCM**, **LBP**, and **HOG**) from banana leaf images, trains a **K-Nearest Neighbors (KNN)** classifier, and serves the prediction through a simple Flask web app.

### ✨ Key Features

* 🧠 Trained KNN model for 3-class classification
* 🎨 Uses advanced image feature extraction (GLCM, LBP, HOG)
* 🧾 Visualizes feature importance and class balance
* 🌐 Flask web interface for uploading and detecting banana leaves
* 🧩 Dataset augmentation and scaling with `MinMaxScaler`

---

## 📂 Project Structure

```
Banana-Leaf-Detector/
│
├── app.py                        # Flask app (main entry)
├── extract_features.py            # Feature extraction functions
├── knn_trainer.py                 # Model training script
├── scale.py                       # Feature scaling and preprocessing
│
├── features_train_banana_aug_balanced.csv   # Extracted feature dataset
├── knn_features_model.pkl         # Trained KNN model
├── label_encoder.pkl              # Encoded class labels
├── scaler.pkl                     # Fitted scaler
│
├── templates/                     # HTML templates for Flask
├── uploads/                       # Temporary uploaded images
├── dataset/                       # Folder for raw and training data (ignored)
│
├── visualization.ipynb            # Jupyter notebook for analysis and plots
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🚀 How to Run the App

1. **Clone the repository**

   ```bash
   git clone https://github.com/Domincee/Banana-Leaf-Detector.git
   cd Banana-Leaf-Detector
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```

5. **Open in your browser**

   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 Model Performance

| Metric  | Training Accuracy | Test Accuracy |
| ------- | ----------------- | ------------- |
| **KNN** | 0.9991            | 0.9127        |

**Classification Report (Test Set)**

* Diseased Leaf → Precision: 0.82 | Recall: 0.92 | F1: 0.87
* Healthy Leaf → Precision: 0.91 | Recall: 0.95 | F1: 0.93
* None-leaf → Precision: 0.98 | Recall: 0.89 | F1: 0.93

---

## 🧬 Dataset Information

The dataset consists of **2,000+ images**, resized to **128×128**, including:

* **Healthy banana leaves**
* **Diseased banana leaves**
* **Non-banana images** (negative samples)

> ⚠️ Raw and training datasets are not included in this repository due to file size limits.
> You can download them from: [[Google Drive Link Here]()](https://drive.google.com/drive/folders/1mng06d0Y_U4hC7WM5hnbBNbuC5ohulcq?usp=sharing)

---

## 🧩 Technologies Used

* **Python 3.11**
* **OpenCV**
* **NumPy & Pandas**
* **scikit-learn**
* **scikit-image**
* **Matplotlib / Seaborn**
* **Flask**

---



