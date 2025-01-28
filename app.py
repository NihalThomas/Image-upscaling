# Import required libraries
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the ESRGAN model path
esrgn_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

@st.cache_resource
def load_model():
    """Load the ESRGAN model from TensorFlow Hub (This will be cached)."""
    return hub.load(esrgn_path)

def preprocess_image(image):
    """Preprocess the input image for the ESRGAN model."""
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    resized_height, resized_width = (height // 4) * 4, (width // 4) * 4
    cropped_image = image_rgb[:resized_height, :resized_width]
    preprocessed_image = cropped_image.astype(np.float32)
    return tf.expand_dims(preprocessed_image, 0)

def run_sr_model(model, image):
    """Run the super-resolution model on the input image."""
    return tf.squeeze(model(image)) / 255.0

# Streamlit application starts here
def main():
    st.title("Super-Resolution Image Enhancement")
    st.write("Upload an image to enhance its resolution using the ESRGAN model.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess image (without caching)
        st.write("Processing the image...")
        model = load_model()  # Load model (this is cached)
        preprocessed_image = preprocess_image(image)  # Do not cache this

        # Run the super-resolution model
        with st.spinner("Enhancing the image..."):
            hr_image = run_sr_model(model, preprocessed_image)  # Do not cache this

        # Display enhanced image
        hr_image = np.clip(hr_image.numpy() * 255, 0, 255).astype(np.uint8)
        st.image(hr_image, caption="Enhanced Image", use_column_width=True)

if __name__ == "__main__":
    main()