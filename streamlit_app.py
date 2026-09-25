import os
import ssl
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Disable SSL verification for Keras weights download on macOS/Cloud
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KERAS_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '.keras'))
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin-top: 0.5rem;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .prediction-card-normal {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 1.5rem;
        border-radius: 8px;
        color: #155724;
        margin-top: 1rem;
    }
    .prediction-card-pneumonia {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 1.5rem;
        border-radius: 8px;
        color: #721c24;
        margin-top: 1rem;
    }
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Deep Learning Model (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_vgg19_model():
    base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(base_model.inputs, output)

    # Check for custom fine-tuned weights
    weights_path = os.path.join(os.path.dirname(__file__), 'vgg_unfrozen.h5')
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            st.toast("Loaded fine-tuned model weights!", icon="✅")
        except Exception as e:
            st.warning(f"Could not load custom weights: {e}")
    return model

# ---------------------------------------------------------
# Image Preprocessing & Prediction
# ---------------------------------------------------------
def predict_image(model, image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(image_bytes).convert('RGB')
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized)
    input_img = np.expand_dims(img_array, axis=0)

    # Model inference
    predictions = model.predict(input_img)
    normal_score = float(predictions[0][0])
    pneumonia_score = float(predictions[0][1])

    label_idx = np.argmax(predictions, axis=1)[0]
    return label_idx, normal_score, pneumonia_score

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🩺 Pneumonia Detection AI Dashboard</h1>
    <p>Automated Chest X-Ray Classification Powered by VGG19 Transfer Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.info("""
    **Architecture**: VGG19 (Transfer Learning)
    **Input Shape**: 128 × 128 × 3
    **Classes**: Normal, Pneumonia
    """)
    
    st.markdown("---")
    st.header("📖 How it Works")
    st.markdown("""
    1. **Upload** a Chest X-Ray image (`.jpg`, `.jpeg`, `.png`).
    2. The image is preprocessed and resized to **128x128**.
    3. **VGG19** extracts deep feature maps.
    4. Custom Dense layers calculate class probabilities.
    """)
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: This tool is for educational & research demonstration purposes only. Consult a certified radiologist for clinical diagnostic decisions.")

# Model Loading
with st.spinner("Initializing Deep Learning Engine (VGG19)..."):
    model = load_vgg19_model()

# Create two columns for Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "Select a chest radiogram image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a standard chest X-ray image for instant analysis."
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-Ray Image", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic Analysis")
    if uploaded_file is not None:
        if st.button("🔍 Run Pneumonia Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing radiogram features..."):
                label_idx, normal_score, pneumonia_score = predict_image(model, uploaded_file)
                
                st.markdown("### **Diagnostic Results**")
                
                if label_idx == 1:
                    st.markdown(f"""
                    <div class="prediction-card-pneumonia">
                        <h2>🔴 Result: PNEUMONIA DETECTED</h2>
                        <p>The model detected opacity/consolidation patterns indicative of pneumonia.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card-normal">
                        <h2>🟢 Result: NORMAL (Healthy)</h2>
                        <p>No significant signs of pneumonia were detected in the X-ray image.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### **Confidence Scores**")
                
                col_n, col_p = st.columns(2)
                with col_n:
                    st.metric("Normal Confidence", f"{normal_score * 100:.1f}%")
                    st.progress(min(normal_score, 1.0))
                with col_p:
                    st.metric("Pneumonia Confidence", f"{pneumonia_score * 100:.1f}%")
                    st.progress(min(pneumonia_score, 1.0))
    else:
        st.info("👈 Please upload a Chest X-Ray image on the left panel to begin diagnostic analysis.")
