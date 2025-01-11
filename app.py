import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

# Page configuration
st.set_page_config(
    page_title="OncoAI - Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define class names (20 classes)
class_names = [
    "Acne/Rosacea", "Actinic Keratosis/Basal Cell Carcinoma", "Atopic Dermatitis",
    "Bullous Disease", "Cellulitis/Impetigo", "Eczema", "Exanthems/Drug Eruptions",
    "Herpes/HPV/STDs", "Light Diseases/Pigmentation Disorders",
    "Lupus/Connective Tissue Diseases", "Melanoma/Skin Cancer/Nevi/Moles",
    "Poison Ivy/Contact Dermatitis", "Psoriasis/Lichen Planus",
    "Seborrheic Keratosis/Benign Tumors", "Systemic Disease",
    "Tinea/Ringworm/Candidiasis/Fungal Infections", "Urticaria Hives",
    "Vascular Tumors", "Vasculitis", "Warts/Molluscum/Viral Infections"
]

# Load model with caching
@st.cache_resource
def load_model():
    try:
        url = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/20skin.pth"
        response = requests.get(url)
        response.raise_for_status()

        # Load pretrained EfficientNet-B0 and modify classifier for 20 classes
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features if isinstance(model.classifier[1], torch.nn.Linear) else model.classifier.in_features
        model.classifier[1] = torch.nn.Linear(num_features, len(class_names))

        state_dict = torch.load(io.BytesIO(response.content), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

model = load_model()

# Preprocess image
def preprocess_image(image):
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)

@torch.no_grad()
def predict(image_tensor):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# Sidebar for Input Method Selection and Image Upload/Capture
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=OncoAI", width=150)  # Replace with your logo URL if available
    st.title("OncoAI")
    st.markdown("### Skin Lesion Classifier")
    st.markdown("Upload or capture an image of a skin lesion to classify it into one of 20 categories.")

    input_method = st.radio("Choose Input Method:", ["Upload Image", "Capture from Camera"])

    img = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a Skin Lesion Image:", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            try:
                img = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Invalid image file: {e}")
    elif input_method == "Capture from Camera":
        camera_image = st.camera_input("Capture a Skin Lesion Image:")
        if camera_image:
            try:
                img_data = camera_image.getvalue()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception as e:
                st.error(f"Invalid camera input: {e}")

# Main Content Area
st.title("🩺 OncoAI - Skin Lesion Classifier")
st.subheader("Detect and Classify Skin Lesions Across 20 Categories")

if img:
    # Display Selected Image in Main Content Area
    st.image(img, caption="Selected Image", use_column_width=True)

    # Analysis and Prediction Section
    with st.spinner("Analyzing the image..."):
        try:
            input_tensor = preprocess_image(img)
            probabilities = predict(input_tensor)

            prediction_idx = np.argmax(probabilities)
            prediction = class_names[prediction_idx]
            confidence_score = probabilities[prediction_idx] * 100

            # Display Predicted Class and Confidence Score
            st.markdown(f"<h3 style='color:#3498db;'>Predicted Class: <strong>{prediction}</strong></h3>", unsafe_allow_html=True)
            st.progress(confidence_score / 100)

            # Display Probabilities with Progress Bars
            st.markdown("<h3>Class Probabilities:</h3>", unsafe_allow_html=True)
            for stage, prob in zip(class_names, probabilities):
                st.markdown(f"<strong>{stage}:</strong> {prob * 100:.2f}%", unsafe_allow_html=True)
                st.progress(prob)

            # Additional Insights Section
            st.markdown("<h3>Additional Insights:</h3>", unsafe_allow_html=True)
            if prediction in ["Melanoma/Skin Cancer/Nevi/Moles", "Actinic Keratosis/Basal Cell Carcinoma"]:
                st.warning(
                    f"The AI detected signs of {prediction}. Please consult a dermatologist for further evaluation."
                )
            else:
                st.success(f"The detected condition ({prediction}) may not be critical. However, consult a dermatologist if symptoms persist.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture a skin lesion image from the sidebar to proceed.")
