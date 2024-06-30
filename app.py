import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('rice_variety_classification_model3.h5')

# Define the class names
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

st.title("Rice Variety Classification")

st.write("Upload an image of rice grains to classify the variety.")

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = Image.open(image)
    img_resized = img.resize((100, 100))
    img_array = np.array(img_resized)
    img_preprocessed = preprocess_input(img_array)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    return img_preprocessed, img_resized

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    preprocessed_image, resized_image = load_and_preprocess_image(uploaded_file)
    
    st.image(resized_image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    
    st.write(f"Prediction: {class_names[predicted_class[0]]}")
    st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")

