import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# Page Configuration
st.set_page_config(
    page_title="OncoAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MODEL_URL = "https://huggingface.co/oculotest/onco/resolve/main/oncosave.h5"
CATEGORIES = [
    "Acne and Rosacea Photos", 
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos", 
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos", 
    "Exanthems and Drug Eruptions", 
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation", 
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles", 
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases", 
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease", 
    "Tinea Ringworm Candidiasis and other Fungal Infections", 
    "Urticaria Hives",
    "Vascular Tumors", 
    "Vasculitis Photos", 
    "Warts Molluscum and other Viral Infections"
]
CONDITION_DESCRIPTIONS = {
    category: f"Description for {category}" for category in CATEGORIES  # Replace with actual descriptions
}
COLORS = {
    category: f"#{np.random.randint(0, 0xFFFFFF):06x}" for category in CATEGORIES  # Random placeholder colors
}

# Preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension and ensure float32

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if not os.path.exists("oncosave.tflite"):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open("oncosave.h5", "wb") as f:
                f.write(response.content)
            
            # Convert to TFLite
            model = tf.keras.models.load_model("oncosave.h5")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            with open("oncosave.tflite", "wb") as f:
                f.write(tflite_model)
            
            os.remove("oncosave.h5")  # Remove the .h5 file to save space
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="oncosave.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

# Prediction function
def predict(image_tensor, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    
    probabilities = interpreter.get_tensor(output_details[0]['index'])
    return probabilities[0]

# Sidebar for Input Method Selection and Image Upload/Capture
with st.sidebar:
    st.header("Input Image")
    input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

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

# Main Content Area for Analysis and Diagnosis
st.title("🩺 OncoAI")
st.subheader("Detect Skin Conditions Across Multiple Categories")
st.markdown("Upload or capture a skin lesion image from the sidebar to analyze potential conditions.")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    interpreter = load_model()

st.success("Model loaded successfully!")

if img:
    # Display Selected Image in Main Content Area
    st.image(img, caption="Selected Image", use_column_width=True)

    # Analysis and Prediction Section
    with st.spinner("Analyzing..."):
        try:
            input_tensor = preprocess_image(img)
            probabilities = predict(input_tensor, interpreter)

            # Display Predicted Category and Description
            prediction_idx = np.argmax(probabilities)
            prediction = CATEGORIES[prediction_idx]
            confidence_score = probabilities[prediction_idx] * 100

            st.markdown(f"<h3 style='color: {COLORS[prediction]}'>Predicted Class: {prediction}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{CONDITION_DESCRIPTIONS[prediction]}</p>", unsafe_allow_html=True)
            st.markdown(f"<strong>Confidence Score:</strong> {confidence_score:.2f}%", unsafe_allow_html=True)

            # Display Probabilities with Progress Bars and Colors
            st.markdown("<h3>Category Probabilities:</h3>", unsafe_allow_html=True)
            for category, prob in zip(CATEGORIES, probabilities):
                st.markdown(f"<strong>{category}:</strong> {prob * 100:.2f}%", unsafe_allow_html=True)
                progress_html = f"""
                <div style="background-color: #e0e0e0; border-radius: 25px; width: 100%; height: 18px; margin-bottom: 10px;">
                    <div style="background-color: {COLORS[category]}; width: {prob * 100}%; height: 100%; border-radius: 25px;"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture a skin lesion image from the sidebar to proceed.")
