import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="OncoAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants for Skin Lesion Detection
MODEL_URL = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/ss_model.pth"
CATEGORIES = ["Benign", "Malignant"]
CONDITION_DESCRIPTIONS = {
    "Benign": "The lesion appears non-cancerous and typically does not pose a threat to health.",
    "Malignant": "The lesion may be cancerous and requires immediate medical attention."
}
COLORS = {"Benign": "#00ff00", "Malignant": "#ff0000"}

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
        return model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

# Grad-CAM visualization
def generate_gradcam(model, input_tensor):
    input_tensor.requires_grad_()
    output = model(input_tensor)
    prediction_idx = output.argmax(dim=1).item()
    class_score = output[0, prediction_idx]
    
    # Backpropagation to get gradients of the score with respect to the last convolutional layer
    model.features[-1].register_forward_hook(lambda m, i, o: setattr(m, 'output', o))
    class_score.backward(retain_graph=True)
    
    gradients = model.features[-1].output.grad
    activations = model.features[-1].output
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations.squeeze(), dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

# Prediction function
@torch.no_grad()
def predict(image_tensor, model):
    image_tensor = image_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# Sidebar for Input Method Selection and Image Upload/Capture
with st.sidebar:
    st.header("Input Image")

    # Clear Data Button
    if st.button("Clear Data"):
        st.session_state.uploader_key += 1  # Increment key to reset file uploader
        st.session_state.current_view = None
        st.experimental_rerun()  # Reload app to apply changes

    # Input Method Selection
    input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

# Main Content Area for Analysis and Diagnosis
st.title("🩺 OncoAI")
st.subheader("Detect Benign or Malignant Skin Lesions")
st.markdown("Upload or capture a skin lesion image from the sidebar to analyze potential conditions.")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    model = load_model()

st.success("Model loaded successfully!")

images = []
if input_method == "Upload Image":
    uploaded_files = st.file_uploader(
        "Upload Skin Lesion Image(s)",
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
    camera_image = st.camera_input("Capture Skin Lesion Image")
    if camera_image:
        try:
            img = Image.open(camera_image).convert("RGB")
            images.append(("Captured Image", img))
        except Exception as e:
            st.error(f"Invalid camera input: {e}")

if images:
    for image_name, img in images:
        st.image(img, caption=f"Selected Image: {image_name}", use_column_width=True)

        with st.spinner(f"Analyzing {image_name}..."):
            try:
                input_tensor = preprocess_image(img)
                probabilities = predict(input_tensor, model)
                prediction_idx = np.argmax(probabilities)
                prediction = CATEGORIES[prediction_idx]
                confidence_score = probabilities[prediction_idx] * 100

                # Display results
                st.markdown(f"<h3 style='color: {COLORS[prediction]}'>Predicted Class: {prediction}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>{CONDITION_DESCRIPTIONS[prediction]}</p>", unsafe_allow_html=True)
                st.markdown(f"<strong>Confidence Score:</strong> {confidence_score:.2f}%", unsafe_allow_html=True)

                # Grad-CAM visualization
                heatmap = generate_gradcam(model, input_tensor)
                plt.imshow(np.asarray(img))
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.axis('off')
                st.pyplot(plt)

            except Exception as e:
                st.error(f"Error during prediction for {image_name}: {e}")
else:
    st.info("Please upload or capture a skin lesion image from the sidebar to proceed.")
