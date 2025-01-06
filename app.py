import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="OncoAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MODEL_URL = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/ss_model.pth"
CATEGORIES = ["Benign", "Malignant"]
COLORS = {"Benign": "#00ff00", "Malignant": "#ff0000"}

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, len(CATEGORIES))
        state_dict = torch.load(io.BytesIO(response.content), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

# Preprocess image with caching
@st.cache_data(show_spinner=False)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Prediction function
@torch.no_grad()
def predict(image_tensor, model):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# UI Header
st.title("🩺 OncoAI")
st.subheader("Detect Benign or Malignant Skin Lesions")
st.markdown("Upload or capture a skin lesion image to analyze potential conditions.")

# Input Method Selection
input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

# Image Input Handling
img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Invalid image file: {e}")
elif input_method == "Capture from Camera":
    camera_image = st.camera_input("Capture Skin Lesion Image")
    if camera_image:
        try:
            img = Image.open(camera_image).convert("RGB")
        except Exception as e:
            st.error(f"Invalid camera input: {e}")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    model = load_model()

# Prediction and Display Results
if img:
    with st.spinner("Analyzing..."):
        try:
            st.image(img, caption="Selected Image", use_column_width=True)
            input_tensor = preprocess_image(img)
            probabilities = predict(input_tensor, model)

            # Display Prediction Results
            prediction_idx = np.argmax(probabilities)
            prediction = CATEGORIES[prediction_idx]
            st.markdown(f"<h3>Predicted Class: {prediction}</h3>", unsafe_allow_html=True)

            # Display Probabilities with Progress Bars
            st.markdown("<h3>Probabilities:</h3>", unsafe_allow_html=True)
            for category, prob in zip(CATEGORIES, probabilities):
                st.markdown(
                    f"<strong>{category}:</strong> {prob * 100:.2f}%",
                    unsafe_allow_html=True,
                )
                progress_html = f"""
                <div style="background-color: #e0e0e0; border-radius: 25px; width: 100%; height: 18px; margin-bottom: 10px;">
                    <div style="background-color: {COLORS[category]}; width: {prob * 100}%; height: 100%; border-radius: 25px;"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture a skin lesion image to proceed.")
