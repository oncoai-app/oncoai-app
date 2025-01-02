import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

st.set_page_config(
    page_title="OculAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="auto",
)

@st.cache_resource
def load_model():
    try:
        url = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/ss_model.pth"
        response = requests.get(url)
        response.raise_for_status()

        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)

        state_dict = torch.load(io.BytesIO(response.content), map_location=torch.device("cpu"))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=(5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
        return probabilities

st.title("OculAI")
st.subheader("One Model, Countless Diseases")

input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
elif input_method == "Capture from Camera":
    camera_image = st.camera_input("Capture Eye Image")
    if camera_image:
        img = Image.open(camera_image).convert("RGB")

if img:
    with st.spinner("Analyzing..."):
        st.image(img, caption="Selected Image", use_column_width=True)

        input_tensor = preprocess_image(img)
        
        try:
            probabilities = predict(input_tensor)

            stages = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative DR (4)"]
            prediction = stages[np.argmax(probabilities)]

            st.markdown(f"<h3>Predicted Stage: {prediction}</h3>", unsafe_allow_html=True)
            
            st.markdown("<h3>Probabilities:</h3>", unsafe_allow_html=True)
            
            for stage, prob in zip(stages, probabilities):
                st.write(f"{stage}: {prob * 100:.2f}%")
                st.progress(prob)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture an eye image to proceed.")
