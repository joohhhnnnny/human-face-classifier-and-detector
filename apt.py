import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
import os

# ───────────────────── PAGE CONFIG ─────────────────────
st.set_page_config(page_title="Human vs Non-Human Detector", page_icon="👤", layout="wide")

# ───────────────────── THEME ─────────────────────
st.markdown("<h1 style='text-align:center;'>Human Face Classifier and Detector</h1>", unsafe_allow_html=True)

# ───────────────────── MODEL SELECTION ─────────────────────
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose AI model:", ["MobileNetV2", "YoloV8"], index=0, key="model_radio"
)

MODEL_PATHS = {
    "MobileNetV2": "./models/mobilenetv2.h5",
    "YoloV8": "./models/best-yolov8s-v2.pt"
}

# ───────────────────── LOAD MODELS ─────────────────────
@st.cache_resource
def load_mobilenet_model(path):
    if not os.path.exists(path):
        st.error(f"MobileNetV2 model not found at:\n{path}")
        st.stop()
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"YOLOv8 model not found at:\n{path}")
        st.stop()
    return YOLO(path)

with st.spinner(f"Loading {model_choice}..."):
    try:
        if model_choice == "MobileNetV2":
            model = load_mobilenet_model(MODEL_PATHS["MobileNetV2"])
        else:
            model = load_yolo_model(MODEL_PATHS["YoloV8"])
    except Exception as e:
        st.error(f"Error loading model:\n{e}")
        st.stop()

# ───────────────────── MOBILENETV2 PREDICTION ─────────────────────
def predict_mobilenet(image_data, model):
    image_data = image_data.convert("RGB")
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

# ───────────────────── YOLO IMAGE DETECTION ─────────────────────
def detect_yolo_image(image, model):
    image_np = np.array(image)
    results = model(image_np)
    return results[0].plot()

# ───────────────────── MAIN UI ─────────────────────
if model_choice == "MobileNetV2":
    file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"], key="mobilenet_uploader")
    if file:
        image = Image.open(file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", width=400)

        if st.button("Analyze Image", key="mobilenet_button"):
            probability = predict_mobilenet(image, model)
            human_prob = 1 - probability
            non_human_prob = probability

            st.metric("Human Probability", f"{human_prob:.2%}")
            st.metric("Non-Human Probability", f"{non_human_prob:.2%}")

            if human_prob > non_human_prob:
                st.success("Result: HUMAN")
            else:
                st.error("Result: NON-HUMAN")

            st.progress(int(human_prob * 100))

else:  # YOLO
    file = st.file_uploader("Upload an image for YOLO detection", type=["jpg", "png", "jpeg"], key="yolo_uploader")
    if file:
        image = Image.open(file).convert("RGB")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Original Image", width=600)

        if st.button("Run YOLO Detection", key="yolo_button"):
            with st.spinner("Detecting..."):
                annotated_image = detect_yolo_image(image, model)
            with col2:
                st.image(annotated_image, caption="YOLO Detection Result", width=600)