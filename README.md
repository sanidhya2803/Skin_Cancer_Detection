# Skin_Cancer_Detection
This project is a Gradio-based web application for detecting Melanoma skin cancer from lesion images using a trained CNN model in TensorFlow. Users can upload an image, and the app classifies it as Benign or Malignant, along with a confidence score. It provides a fast, user-friendly interface for early skin cancer risk assessment.

# 🧪 Melanoma Skin Cancer Classifier

This is a deep learning-based web application that detects **Melanoma (skin cancer)** from images of skin lesions using a trained Convolutional Neural Network (CNN) model. The app is built using **Gradio** and leverages **TensorFlow** for model inference.

---

## 🚀 Features

- Upload a skin lesion image
- Predict whether the lesion is **Benign** or **Malignant**
- Displays classification result and prediction confidence
- Simple, intuitive web interface using Gradio

---

## 🧠 Model Information

- Model file: `melanoma_classification_model.h5`
- Input image size: **150x150**
- Output: Binary classification (Benign or Malignant)

---

## 🛠 Installation & Usage

### 1. Clone the repository

bash
git clone https://github.com/your-username/melanoma-skin-cancer-classifier.git
cd melanoma-skin-cancer-classifier

### 2. Install Dependencies
Make sure you are in a virtual environment, then run:
bash
Copy
Edit
pip install -r requirements.txt

### 3. Run the App
bash
Copy
Edit
python app.py

