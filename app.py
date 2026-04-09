import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st

# ================================
# MUST BE FIRST STREAMLIT COMMAND
# ================================
st.set_page_config(
    page_title="Knee X-Ray Classifier",
    page_icon="🦴",
    layout="centered"
)

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ================================
# Build Exact Model Architecture
# ================================
@st.cache_resource
def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.load_weights("model.h5", by_name=True, skip_mismatch=True)
    return model

model = get_model()

# ================================
# Class Names
# ================================
class_names = ["Normal", "Osteopenia", "Osteoporosis"]

# ================================
# App Header
# ================================
st.title("🦴 Knee Osteoporosis X-Ray Classifier")
st.markdown("""
This app analyzes knee X-ray images and classifies them into:
- ✅ **Normal** — Healthy knee
- ⚠️ **Osteopenia** — Low bone density
- 🚨 **Osteoporosis** — Severely low bone density
""")
st.divider()

# ================================
# Sidebar
# ================================
st.sidebar.title("📊 Model Info")
st.sidebar.success("✅ Model Loaded Successfully")
st.sidebar.write(f"**Input Shape:** {model.input_shape}")
st.sidebar.write(f"**Total Layers:** {len(model.layers)}")
st.sidebar.divider()
st.sidebar.write("**Class Labels:**")
st.sidebar.write("1 → Normal")
st.sidebar.write("1 → Osteopenia")
st.sidebar.write("1 → Osteoporosis")

# ================================
# File Uploader
# ================================
uploaded_file = st.file_uploader(
    "📁 Upload a Knee X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a knee X-ray image in JPG or PNG format"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded X-Ray Image", use_column_width=True)
    st.divider()

    # ================================
    # Preprocess — NO /255 division!
    # Matches training exactly
    # ================================
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ================================
    # Predict
    # ================================
    with st.spinner("🔍 Analyzing X-Ray... Please wait"):
        prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction[0]))

    # ================================
    # Show Result
    # ================================
    st.subheader("🎯 Prediction Result")

    if predicted_class == "Normal":
        st.success(f"## ✅ {predicted_class}")
        st.write("The knee appears **healthy** with normal bone density.")
    elif predicted_class == "Osteopenia":
        st.warning(f"## ⚠️ {predicted_class}")
        st.write("The knee shows signs of **low bone density**. Medical consultation recommended.")
    else:
        st.error(f"## 🚨 {predicted_class}")
        st.write("The knee shows signs of **severely low bone density**. Please consult a doctor immediately.")

    st.metric(label="Confidence Score", value=f"{confidence:.2%}")

    # ================================
    # All Probabilities
    # ================================
    st.divider()
    st.subheader("📊 All Class Probabilities")
    for cls, prob in zip(class_names, prediction[0]):
        prob_float = float(prob)
        st.write(f"**{cls}:** {prob_float:.2%}")
        st.progress(prob_float)

    # Debug in sidebar
    st.sidebar.divider()
    st.sidebar.write("### 🔢 Raw Scores:")
    for cls, prob in zip(class_names, prediction[0]):
        st.sidebar.write(f"{cls}: {float(prob):.6f}")

    # ================================
    # Disclaimer
    # ================================
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for **educational purposes only** "
        "and does not replace professional medical advice. "
        "Always consult a qualified medical professional for diagnosis."
    )