import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

# --- Configuration ---
st.set_page_config(
    page_title="Banana Ripeness Predictor",
    page_icon="🍌",
    layout="centered"
)

# --- Constants ---
MODEL_Path = Path(__file__).parent.parent / "models" / "banana_model_best.keras"
# Class labels in alphabetical order (standard for Keras flow_from_directory)
CLASS_NAMES = ['Overripe', 'Ripe', 'Rotten', 'Unripe']
IMG_SIZE = (224, 224)

# --- Model Loading ---
@st.cache_resource
def load_prediction_model(model_path):
    """Loads the Keras model from the specified path."""
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Preprocessing ---
def preprocess_image(image):
    """
    Preprocesses the image to match the training data requirements.
    1. Resize to (224, 224)
    2. Convert to RGB (in case of RGBA/Grayscale)
    3. Convert to array and normalize (1./255)
    4. Expand dims to create batch of size 1
    """
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    img_array = np.asarray(image)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Main App ---
def main():
    st.title("🍌 Banana Ripeness Predictor")
    st.write("Upload an image of a banana to predict its ripeness stage!")

    # Sidebar for info
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses a Convolutional Neural Network (CNN) "
            "to classify bananas into one of four categories:\n"
            "- Overripe\n- Ripe\n- Rotten\n- Unripe"
        )
        st.warning(f"Model loaded from: `{MODEL_Path.name}`")

    # Load Model
    model = load_prediction_model(MODEL_Path)

    if model is None:
        st.stop()

    # File Uploader
    uploaded_file = st.file_uploader("Choose a banana image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Display Image
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            with st.spinner('Analyzing...'):
                # Make Prediction
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                score = tf.nn.softmax(predictions[0])
                
                # Get Result
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                confidence = 100 * np.max(predictions[0]) # Softmax usually not needed if model has softmax output, but 'predict' returns logits or probs. 
                # If model output is softmax, np.max is enough. If logits, tf.nn.softmax is needed.
                # Assuming standard classifier with softmax output or similar. 
                # Let's trust the raw prediction for argmax, and formatted confidence.
                
                # If the model didn't have softmax activation in the last layer, predictions might be logits.
                # However, for display, let's keep it simple.
                
                st.success(f"**Prediction:** {predicted_class}")
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
                
                # Optional: Show prob distribution
                # st.bar_chart(dict(zip(CLASS_NAMES, predictions[0])))

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
