# 🩺 Pneumonia Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)](https://flask.palletsprojects.com/)
[![VGG19](https://img.shields.io/badge/Architecture-VGG19-red.svg)](https://keras.io/api/applications/vgg/#vgg19-function)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning solution for automated detection of Pneumonia from Chest X-Ray images. This project utilizes transfer learning with a fine-tuned **VGG19** neural network architecture integrated into an intuitive **Flask web application** for real-time medical image diagnosis.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Training the Model](#1-training-the-model)
  - [Running the Web App](#2-running-the-web-app)
- [Results & Output](#-results--output)
- [License](#-license)

---

## 🔍 Overview
Pneumonia is a life-threatening inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. Early and precise diagnosis from chest radiograms is critical for effective treatment. 

This project addresses the challenge by leveraging Deep Convolutional Neural Networks (CNNs). Using **VGG19** as a pre-trained feature extractor combined with custom classification layers, the model classifies chest X-ray images into two classes:
- **Normal**: Healthy lung condition.
- **Pneumonia**: Presence of lung infection/inflammation.

---

## ✨ Key Features
- **High-Accuracy Classification**: Utilizes VGG19 deep learning architecture fine-tuned on chest X-ray datasets.
- **Interactive Web UI**: Responsive web app built with Flask, Bootstrap, and jQuery for uploading X-ray images and getting instant diagnostic results.
- **Real-Time Processing**: Pre-processes uploaded X-rays (resizing to $128 \times 128$, RGB conversion) and outputs predictions in seconds.
- **Complete ML Pipeline**: Includes data preprocessing, model evaluation, and deployment scripts.

---

## 🛠 Tech Stack

| Domain | Technologies / Libraries |
| :--- | :--- |
| **Deep Learning** | TensorFlow, Keras, VGG19 |
| **Data & Image Processing** | OpenCV, NumPy, Pillow (PIL), Matplotlib |
| **Backend Framework** | Python, Flask, Werkzeug |
| **Frontend UI** | HTML5, CSS3, JavaScript, jQuery, Bootstrap 4 |
| **Environment** | Jupyter Notebook, Python 3.8+ |

---

## 🧠 Model Architecture

The transfer learning model is built upon the **VGG19** backbone (pre-trained on ImageNet):

```text
Input (128x128x3 RGB Image)
        │
        ▼
   VGG19 Base (Feature Extractor)
        │
        ▼
 Flatten Layer
        │
        ▼
 Dense Layer (4608 units, ReLU)
        │
        ▼
 Dropout Layer (0.2 rate)
        │
        ▼
 Dense Layer (1152 units, ReLU)
        │
        ▼
 Dense Output Layer (2 units, Softmax)
        │
        ▼
 Predictions: [Normal / Pneumonia]
```

---

## 📁 Repository Structure

```text
PNEUMONIA_DETECTION/
├── Flask Application/
│   ├── app.py                   # Main Flask server script
│   ├── templates/               # HTML template files
│   │   ├── import.html          # Base layout template
│   │   └── index.html           # Main upload & prediction dashboard
│   ├── static/                  # CSS, JS, and vendor styling libraries
│   │   ├── css/                 # Bootstrap & custom styling
│   │   └── js/                  # jQuery & custom prediction scripts
│   └── uploads/                 # Storage directory for user-uploaded X-rays
├── Pneumonia Detection Using Deep Learning.ipynb  # Model building & training notebook
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- **Python 3.8+**
- **pip** (Python package installer)
- **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/proswarnali24/PNEUMONIA_DETECTION.git
   cd PNEUMONIA_DETECTION
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Training the Model
To re-train or experiment with the model architecture:
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Pneumonia Detection Using Deep Learning.ipynb`.
3. Follow the notebook steps to load the dataset, train the VGG19 model, evaluate performance, and export the trained model weights (`vgg_unfrozen.h5`).

### 2. Running the Web App
1. Place your trained model weights file (`vgg_unfrozen.h5`) inside the `Flask Application/` directory (or generate it using the notebook).
2. Navigate to the `Flask Application` folder:
   ```bash
   cd "Flask Application"
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```
5. Upload a Chest X-ray image (`.jpg`, `.jpeg`, `.png`) and click **Predict!** to view the classification result.

---

## 📊 Results & Output

The application provides real-time classification feedback:
- **Normal**: Indicates no signs of pneumonia in the uploaded chest radiogram.
- **Pneumonia**: Indicates detected opacities/consolidation consistent with pneumonia.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---

<p center>Crafted with ❤️ for AI in Healthcare</p>