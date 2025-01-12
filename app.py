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

# Disease configurations
DISEASE_CONFIGS = {
    "Skin Cancer": {
        "MODEL_URL": "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/ss_model.pth",
        "CATEGORIES": ["Benign", "Malignant"],
        "CONDITION_DESCRIPTIONS": {
            "Benign": "The lesion appears non-cancerous and typically does not pose a threat to health.",
            "Malignant": "The lesion may be cancerous and requires immediate medical attention."
        },
        "UPLOAD_TITLE": "Upload Skin Lesion Image(s)",
        "CAMERA_TITLE": "Capture Skin Lesion Image",
        "SUBTITLE": "Upload or capture a skin lesion image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction} growth. Please consult a dermatologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a skin lesion image from the sidebar to proceed."
    },
    "Breast Cancer": {
        "MODEL_URL": "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/bc_mammogram.pth",
        "CATEGORIES": ["Benign", "Malignant"],
        "CONDITION_DESCRIPTIONS": {
            "Benign": "The lesion appears non-cancerous and is unlikely to pose a threat to health, but may require routine monitoring.",
            "Malignant": "The lesion may be cancerous and requires immediate medical evaluation and further testing."
        },
        "UPLOAD_TITLE": "Upload Mammogram(s)",
        "CAMERA_TITLE": "Capture Mammogram Image",
        "SUBTITLE": "Upload or capture a mammogram image from the sidebar to analyze potential conditions.",
        "WARNING_MESSAGE": "The AI detected signs of {prediction} growth. Please consult an oncologist for further evaluation.",
        "INFO_MESSAGE": "Please upload or capture a mammogram image from the sidebar to proceed."
    }
}

# Sidebar for disease selection
with st.sidebar:
    st.header("Select Disease")
    selected_disease = st.selectbox("Choose a disease to analyze:", list(DISEASE_CONFIGS.keys()))

# Load selected disease configuration
config = DISEASE_CONFIGS[selected_disease]
MODEL_URL = config["MODEL_URL"]
CATEGORIES = config["CATEGORIES"]
CONDITION_DESCRIPTIONS = config["CONDITION_DESCRIPTIONS"]

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

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model(model_url):
    try:
        response = requests.get(model_url)
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

# Prediction function
@torch.no_grad()
def predict(image_tensor, model):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# Initialize session state for file uploader key and current view
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if 'current_view' not in st.session_state:
    st.session_state.current_view = None

# Sidebar for Input Method Selection and Image Upload/Capture
with st.sidebar:
    st.header("Input Image")

    # Display currently viewed image at the top of the sidebar
    if st.session_state.current_view:
        st.image(st.session_state.current_view[1], caption=st.session_state.current_view[0], use_column_width=True)
        st.markdown("---")

    # Clear Data Button
    if st.button("Clear Data"):
        st.session_state.uploader_key += 1  # Increment key to reset file uploader
        st.session_state.current_view = None
        st.experimental_rerun()  # Reload app to apply changes

    # Input Method Selection
    input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

    images = []
    if input_method == "Upload Image":
        uploaded_files = st.file_uploader(
            config["UPLOAD_TITLE"],
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    images.append((uploaded_file.name, img))
                except Exception as e:
                    st.error(f"Invalid image file: {e}")
    elif input_method == "Capture from Camera":
        camera_image = st.camera_input(config["CAMERA_TITLE"])
        if camera_image:
            try:
                img = Image.open(camera_image).convert("RGB")
                images.append(("Captured Image", img))
            except Exception as e:
                st.error(f"Invalid camera input: {e}")

# Main Content Area for Analysis and Diagnosis
st.title("🩺 OncoAI")
st.subheader("Detect Benign or Malignant Masses")
st.markdown("Upload or capture a mammogram image from the sidebar to analyze potential conditions.")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    model = load_model()

st.success("Model loaded successfully!")

if images:
    # Single image upload
    if len(images) == 1:
        image_name, img = images[0]
        st.image(img, caption=f"Selected Image: {image_name}", use_column_width=True)

        # Analysis and Prediction Section
        with st.spinner(f"Analyzing {image_name}..."):
            try:
                # Preprocess image and get predictions
                input_tensor = preprocess_image(img)
                probabilities = predict(input_tensor, model)
                prediction_idx = np.argmax(probabilities)
                prediction = CATEGORIES[prediction_idx]
                confidence_score = probabilities[prediction_idx] * 100

                # Display detailed results for the single image
                st.markdown(f"<h3 style='color: {COLORS[prediction]}'>Predicted Class: {prediction}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>{CONDITION_DESCRIPTIONS[prediction]}</p>", unsafe_allow_html=True)
                st.markdown(f"<strong>Confidence Score:</strong> {confidence_score:.2f}%", unsafe_allow_html=True)

                # Display category probabilities with progress bars
                st.markdown("<h3>Category Probabilities:</h3>", unsafe_allow_html=True)
                for category, prob in zip(CATEGORIES, probabilities):
                    st.markdown(f"<strong>{category}:</strong> {prob * 100:.2f}%", unsafe_allow_html=True)
                    progress_html = f"""
                    <div style="background-color: #e0e0e0; border-radius: 25px; width: 100%; height: 18px; margin-bottom: 10px;">
                        <div style="background-color: {COLORS[category]}; width: {prob * 100}%; height: 100%; border-radius: 25px;"></div>
                    </div>
                    """
                    st.markdown(progress_html, unsafe_allow_html=True)

                # Additional insights or warnings based on prediction
                if prediction != "Benign":
                    st.warning(config["WARNING_MESSAGE"].format(prediction=prediction))
                else:
                    st.success("The analysis shows no concerning signs.")
            except Exception as e:
                st.error(f"Error during prediction for {image_name}: {e}")

    # Multiple image uploads
    else:
        st.markdown("### Uploaded Images and Predictions")
        for image_name, img in images:
            col1, col2, col3 = st.columns([8, 1, 1])

            # Show spinner while analyzing each image sequentially
            with st.spinner(f"Analyzing {image_name}..."):
                try:
                    # Preprocess image and get predictions
                    input_tensor = preprocess_image(img)
                    probabilities = predict(input_tensor, model)
                    prediction_idx = np.argmax(probabilities)
                    prediction = CATEGORIES[prediction_idx]
                    confidence_score = probabilities[prediction_idx] * 100

                    # Display results in columns
                    with col1:
                        st.markdown(
                            f"**{image_name}**: <span style='color:{COLORS[prediction]}'>{prediction}</span> ({confidence_score:.2f}%)",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        if st.button("View", key=f"view_btn_{image_name}"):
                            st.session_state.current_view = (image_name, img)
                    with col3:
                        if st.button("✕", key=f"close_btn_{image_name}"):
                            if st.session_state.current_view and st.session_state.current_view[0] == image_name:
                                st.session_state.current_view = None

                except Exception as e:
                    st.error(f"Error during prediction for {image_name}: {e}")

else:
    st.info("Please upload or capture a mammogram image from the sidebar to proceed.")
