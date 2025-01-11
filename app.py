import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="OncoAI - Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide",
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

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        
        class_idx = output.argmax(dim=1)
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()

# Load model with caching
@st.cache_resource
def load_model():
    try:
        url = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/20skin.pth"
        response = requests.get(url)
        response.raise_for_status()

        # Load pretrained EfficientNet-B0 and modify classifier for 20 classes
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, len(class_names))

        state_dict = torch.load(io.BytesIO(response.content), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

model = load_model()

# Preprocess image with advanced data augmentation
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

# Grad-CAM visualization function
def generate_grad_cam(image_tensor):
    target_layer = model.features[-1]  # Last convolutional layer in EfficientNet-B0
    cam = GradCAM(model=model, target_layer=target_layer)
    
    grayscale_cam = cam(image_tensor)
    rgb_image = image_tensor.squeeze().permute(1, 2, 0).numpy()
    rgb_image = (rgb_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + rgb_image
    cam_image /= np.max(cam_image)
    
    return np.uint8(255 * cam_image)

# Streamlit UI setup
st.title("🩺 OncoAI - Skin Lesion Classifier")
st.subheader("Detect and Classify Skin Lesions Across 20 Categories")

# Input method selection
input_method = st.radio("Choose Input Method:", ["Upload Image", "Capture from Camera"])

img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a Skin Lesion Image:", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
elif input_method == "Capture from Camera":
    camera_image = st.camera_input("Capture a Skin Lesion Image:")
    if camera_image:
        img = Image.open(camera_image).convert("RGB")

if img:
    with st.spinner("Analyzing the image..."):
        st.image(img, caption="Selected Image", use_column_width=True)

        input_tensor = preprocess_image(img)
        
        try:
            probabilities = predict(input_tensor)

            prediction_idx = np.argmax(probabilities)
            prediction = class_names[prediction_idx]

            # Display predictions and probabilities with enhanced styling
            st.markdown(f"<h3 style='color:#3498db;'>Predicted Class: <strong>{prediction}</strong></h3>", unsafe_allow_html=True)
            
            st.markdown("<h3>Class Probabilities:</h3>", unsafe_allow_html=True)

            for stage, prob in zip(class_names, probabilities):
                st.write(f"{stage}: {prob * 100:.2f}%")
            
            # Grad-CAM visualization remains unchanged
            
            cam_image = generate_grad_cam(input_tensor)
            st.image(cam_image, caption="Grad-CAM Visualization (Highlighting Key Areas)", use_column_width=True)
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture a skin lesion image to proceed.")
