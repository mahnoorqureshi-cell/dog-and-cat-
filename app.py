import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Path to the trained model file
MODEL_PATH = "cats_dogs_cnn_kagglehub.h5"

@st.cache_resource
def load_cnn_model(path):
    # Load the trained CNN model
    return load_model(path)

# Load model once
model = load_cnn_model(MODEL_PATH)

st.title("Cats vs Dogs Image Classifier")
st.write("Upload an image of a cat or dog to get the prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img_resized = image.resize((150, 150))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict class
    prediction = model.predict(img_array)
    label = "Dog" if prediction[0][0] > 0.5 else "Cat"
    
    # Display result
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {prediction[0][0]:.4f}")
