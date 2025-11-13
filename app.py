# Paste all your Streamlit code below
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

MODEL_PATH = "cats_dogs_cnn_kagglehub.h5"

@st.cache_resource
def load_cnn_model(path):
    return load_model(path)

model = load_cnn_model(MODEL_PATH)

st.title("Cats vs Dogs Classifier")
uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((150,150))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    label = "Dog" if prediction[0][0] > 0.5 else "Cat"
    st.write("Prediction:", label)
    st.write("Confidence:", prediction[0][0])
