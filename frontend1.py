import streamlit as st
import numpy as np
import cv2
import joblib  # For loading trained models
from PIL import Image
import logging  # Import logging

# Configure logging (optional, but recommended)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model (Change filename based on the model you want to use)
try:
    model = joblib.load("decision_tree.pkl")  # Update based on actual model filename
    st.success("Model loaded successfully!") # Display a success message upon loading
except Exception as e:
    st.error(f"Error loading model: {e}")
    logging.error(f"Error loading model: {e}")
    model = None  # Ensure model is None if loading fails

# Class labels for FER2013 dataset
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(image):
    """Preprocess uploaded image for model prediction."""
    try:
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((48, 48))  # Resize to match training data
        image_array = np.array(image).flatten() / 255.0  # Flatten and normalize
        return image_array.reshape(1, -1)  # Reshape for model
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        logging.error(f"Error preprocessing image: {e}")
        return None  # Return None to indicate failure

# Streamlit UI
st.title("Facial Emotion Recognition")
st.write("Upload an image to detect emotions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Preprocess image
        processed_image = preprocess_image(image)

        if processed_image is not None:
            # Make prediction
            if model is not None:  # Check if the model loaded successfully
                try:
                    prediction = model.predict(processed_image)
                    predicted_label = class_names[int(prediction[0])]
                    st.write(f"**Predicted Emotion:** {predicted_label}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    logging.error(f"Error during prediction: {e}")
            else:
                st.error("Model loading failed.  Cannot make predictions.")
        else:
            st.error("Image preprocessing failed.")
    except Exception as e:
        st.error(f"Error opening the image: {e}")
        logging.error(f"Error opening the image: {e}")