# 🩺 Pneumonia Detection AI Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pneumoniadetection-uqy2vdrv7t4djwyx3qcd6q.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![VGG19](https://img.shields.io/badge/Architecture-VGG19-red.svg)](https://keras.io/api/applications/vgg/#vgg19-function)

An end-to-end Computer Vision and Deep Learning web dashboard for real-time automated detection of **Pneumonia** from Chest X-Ray radiograms. Built using **Streamlit** and **VGG19 Transfer Learning**, this application enables rapid diagnostic screening with visual confidence scoring.

---

## 🌐 Live Web Application

Access the live application directly in your browser:
👉 **[https://pneumoniadetection-uqy2vdrv7t4djwyx3qcd6q.streamlit.app/](https://pneumoniadetection-uqy2vdrv7t4djwyx3qcd6q.streamlit.app/)**

---

## ✨ Features

- **⚡ Instant Classification**: Analyzes chest radiograms in seconds and classifies them as **Normal** or **Pneumonia**.
- **🧠 VGG19 Backbone**: Uses a deep convolutional transfer learning network trained on chest X-Ray datasets.
- **📊 Interactive Metrics**: Real-time confidence percentage scores and visual probability bars for both diagnostic classes.
- **🎨 Modern Responsive UI**: Custom-styled Streamlit interface featuring a dark medical header, upload card, and status badges.
- **📁 Included Test Dataset**: Sample X-ray images included in `samples/` for immediate testing.

---

## 🛠 Tech Stack

| Component | Technologies / Libraries |
| :--- | :--- |
| **Web Interface** | Streamlit, HTML5, CSS3 |
| **Deep Learning** | TensorFlow, Keras (VGG19 Backbone) |
| **Image Preprocessing** | OpenCV, Pillow (PIL), NumPy |
| **Model Exploration** | Jupyter Notebook |

---

## 🧠 Deep Learning Architecture

The underlying neural network uses the **VGG19** model pre-trained on ImageNet as a feature extractor, attached to a custom classification head:

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
 Predictions: [Normal (0) / Pneumonia (1)]
```

---

## 📁 Repository Structure

```text
PNEUMONIA_DETECTION/
├── app.py                                       # Main Streamlit application
├── streamlit_app.py                             # Streamlit Cloud deployment entrypoint
├── Pneumonia Detection Using Deep Learning.ipynb  # Model building & training notebook
├── samples/                                     # Test X-Ray dataset
│   ├── test_chest_xray_1.jpg
│   ├── test_normal_xray.jpeg
│   └── test_pneumonia_xray.jpeg
├── requirements.txt                             # Python dependencies for Streamlit
├── .gitignore                                   # Ignore cache, venv, and IDE files
└── README.md                                    # Project documentation
```

---

## 💻 Local Installation & Setup

### Prerequisites
- **Python 3.8+**
- **pip**

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/proswarnali24/PNEUMONIA_DETECTION.git
   cd PNEUMONIA_DETECTION
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🧪 Testing with Sample Images

Sample X-ray radiograms are provided in the [`samples/`](samples/) folder for immediate testing:
- **`samples/test_normal_xray.jpeg`**: Healthy chest radiogram.
- **`samples/test_pneumonia_xray.jpeg`**: Pneumonia chest radiogram.
- **`samples/test_chest_xray_1.jpg`**: High-resolution chest radiogram.

---

## ⚠️ Disclaimer

This application is created for **educational, demonstration, and research purposes only**. It should not be used as a primary diagnostic tool for clinical medical decisions without validation by a certified medical professional or radiologist.