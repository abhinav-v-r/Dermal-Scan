import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import traceback
import pandas as pd
import io
import base64
import csv
import datetime
import logging

# Define class labels
class_labels = ['clear_face', 'dark_spots', 'puffy_eyes', 'wrinkles']

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"dermal_scan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper functions for export
def get_csv_download_link(df, filename="prediction_results.csv"):
    """Generate a download link for a CSV file from a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Results</a>'
    return href

def get_image_download_link(img, filename="annotated_image.jpg", text="Download Annotated Image"):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    img_pil = Image.fromarray(img)
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Set up error handling
@st.cache_resource
def load_models():
    """Load all required models with error handling"""
    models = {
        "skin_model": None,
        "faceNet": None,
        "ageNet": None,
        "face_cascade": None
    }
    
    try:
        # Try to import TensorFlow
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Load the skin condition model
        model_path = os.path.join(os.path.dirname(__file__), "models", "dherma_ai_scan_v1.h5")
        if os.path.exists(model_path):
            try:
                # Handle the multiple input tensors issue
                import tensorflow as tf
                
                # Custom loading approach for models with multiple input tensors
                # First, load the model architecture without weights
                model_json = os.path.join(os.path.dirname(__file__), "models", "model_architecture.json")
                
                # If model architecture JSON exists, use it
                if os.path.exists(model_json):
                    with open(model_json, 'r') as f:
                        model_config = f.read()
                    models["skin_model"] = tf.keras.models.model_from_json(model_config)
                    # Load weights separately
                    models["skin_model"].load_weights(model_path)
                    st.sidebar.success("✅ Skin condition model loaded successfully from architecture + weights")
                else:
                    # Try loading with custom function to handle the input tensor issue
                    def load_model_with_concat(model_path):
                        # Load the base model
                        base_model = tf.keras.applications.DenseNet121(
                            input_shape=(224, 224, 3),
                            include_top=False,
                            weights=None
                        )
                        
                        # Get the output from the base model
                        x = base_model.output
                        
                        # Add a global spatial average pooling layer
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        
                        # Add a fully-connected layer
                        x = tf.keras.layers.Dense(128, activation='relu')(x)
                        
                        # Add a logistic layer for predictions
                        predictions = tf.keras.layers.Dense(len(class_labels), activation='softmax')(x)
                        
                        # Create the model
                        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                        
                        # Try to load weights from the saved model
                        try:
                            temp_model = tf.keras.models.load_model(model_path, compile=False)
                            # Get weights from the dense and output layers
                            dense_weights = None
                            output_weights = None
                            
                            for layer in temp_model.layers:
                                if isinstance(layer, tf.keras.layers.Dense):
                                    if layer.name == 'dense':
                                        dense_weights = layer.get_weights()
                                    elif layer.name == 'dense_1' or layer.name.endswith('predictions'):
                                        output_weights = layer.get_weights()
                            
                            # Set weights to our new model if found
                            if dense_weights is not None and output_weights is not None:
                                model.layers[-2].set_weights(dense_weights)
                                model.layers[-1].set_weights(output_weights)
                        except Exception as weight_e:
                            logging.warning(f"Could not transfer weights: {str(weight_e)}")
                            
                        return model
                    
                    # Try loading with our custom function
                    models["skin_model"] = load_model_with_concat(model_path)
                    st.sidebar.success("✅ Skin condition model loaded successfully with custom loader")
                
                # Log successful model loading
                logging.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                # Log the error for debugging
                error_msg = f"Error loading model: {str(e)}"
                logging.error(error_msg)
                st.sidebar.error(f"❌ {error_msg}")
                
                # Display more helpful information to the user
                st.sidebar.error("The model trained in Colab has multiple input tensors issue.")
                st.sidebar.info("💡 In Colab, save your model architecture: with open('model_architecture.json', 'w') as f: f.write(model.to_json())")
                st.sidebar.info("💡 Then save weights: model.save_weights('model_weights.h5')")
                
                # Don't create an alternative model with random weights as it will give incorrect predictions
                models["skin_model"] = None
        else:
            error_msg = f"Model file not found: {model_path}"
            logging.error(error_msg)
            st.sidebar.error(f"❌ {error_msg}")
            st.sidebar.info("💡 Make sure to copy your trained model from Colab to the models directory")
            
        # Paths for age prediction model
        faceProto = os.path.join(os.path.dirname(__file__), "models", "age_prediction", "opencv_face_detector.pbtxt")
        faceModel = os.path.join(os.path.dirname(__file__), "models", "age_prediction", "opencv_face_detector_uint8.pb")
        ageProto = os.path.join(os.path.dirname(__file__), "models", "age_prediction", "age_deploy.prototxt")
        ageModel = os.path.join(os.path.dirname(__file__), "models", "age_prediction", "age_net.caffemodel")
        
        # Check if all files exist
        files_exist = all(os.path.exists(f) for f in [faceProto, faceModel, ageProto, ageModel])
        
        if files_exist:
            # Load age prediction models
            models["faceNet"] = cv2.dnn.readNet(faceModel, faceProto)
            models["ageNet"] = cv2.dnn.readNet(ageModel, ageProto)
            st.sidebar.success("✅ Age prediction models loaded successfully")
        else:
            missing = [f for f in [faceProto, faceModel, ageProto, ageModel] if not os.path.exists(f)]
            st.sidebar.warning(f"⚠️ Some age prediction model files not found: {', '.join(missing)}")
        
    except ImportError as e:
        st.sidebar.error(f"❌ Error importing TensorFlow: {str(e)}")
       
    except Exception as e:
       
        st.sidebar.text(traceback.format_exc())
    
    # Load face cascade (OpenCV built-in)
    try:
        models["face_cascade"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
       
    except Exception as e:
        st.sidebar.error(f"❌ Error loading face cascade: {str(e)}")
    
    return models

# Load all models
models = load_models()

# Age ranges
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Function to predict age with fallback
def predict_age(face):
    try:
        if models["ageNet"] is None:
            return "Unknown"
            
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        models["ageNet"].setInput(blob)
        agePreds = models["ageNet"].forward()
        age = ageList[agePreds[0].argmax()]
        return age
    except Exception as e:
        st.warning(f"⚠️ Age prediction error: {str(e)}")
        return "Unknown"

# Function to predict skin conditions with fallback
def predict_skin_condition(face_img, threshold=0.5):  # Increased threshold for better accuracy
    try:
        if models["skin_model"] is None:
            return {"Model not loaded": 100}
            
        # Resize and preprocess image for MobileNetV2 (standard preprocessing)
        img = cv2.resize(face_img, (224, 224))
        
        # Convert BGR to RGB (OpenCV uses BGR, but most models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply MobileNetV2 preprocessing (scale to [-1, 1] range)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 127.5 - 1.0  # Scale to [-1, 1] instead of [0, 1]
        img_array = np.expand_dims(img_array, axis=0)
        
        # Log preprocessing for debugging
        logging.info(f"Image shape: {img_array.shape}, Range: [{img_array.min()}, {img_array.max()}]")
        
        # Predict
        predictions = models["skin_model"].predict(img_array)[0]
        results = dict(zip(class_labels, predictions))
        
        # Filter based on threshold
        filtered_results = {cls: round(prob * 100, 2) for cls, prob in results.items() if prob >= threshold}
        
        # If no conditions above threshold, return the highest one
        if not filtered_results:
            top_condition = max(results.items(), key=lambda x: x[1])
            filtered_results = {top_condition[0]: round(top_condition[1] * 100, 2)}
            
        return filtered_results
    except Exception as e:
        st.warning(f"⚠️ Skin condition prediction error: {str(e)}")
        return {"Error": 100}

# Combined prediction function with error handling
def predict_face_features(face_img):
    age = predict_age(face_img)
    conditions = predict_skin_condition(face_img)
    return age, conditions

# Streamlit UI
st.set_page_config(page_title="Dhermal AI Scan", layout="centered")
st.title("🧑‍⚕️ Dhermal AI Scan – Face & Age Prediction")
st.write("Upload a face image to predict **Age** and detect **Skin Conditions**.")

# Add sidebar info
st.sidebar.title("ℹ️ Model Status")
st.sidebar.write("Check model loading status below:")

# Add sample images section in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Test with Sample Images")
sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
if os.path.exists(sample_dir):
    sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if sample_images:
        selected_sample = st.sidebar.selectbox(
            "Select a sample image to test:",
            ["None"] + sample_images
        )
        
        if selected_sample != "None":
            st.sidebar.success(f"Using sample image: {selected_sample}")
            logging.info(f"Testing with sample image: {selected_sample}")
else:
    st.sidebar.warning("Sample images directory not found")

# Upload button
uploaded_file = st.file_uploader(
    "👉 Upload a Face Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

# Process either uploaded file or selected sample
image_to_process = None

if uploaded_file is not None:
    image_to_process = uploaded_file
    logging.info(f"Processing uploaded image")
elif 'selected_sample' in locals() and selected_sample != "None":
    sample_path = os.path.join(sample_dir, selected_sample)
    if os.path.exists(sample_path):
        image_to_process = sample_path
        logging.info(f"Processing sample image: {sample_path}")

if image_to_process is not None:
    try:
        # Load and show image (either uploaded or sample)
        if isinstance(image_to_process, str):  # Sample image path
            image = Image.open(image_to_process).convert("RGB")
            st.image(image, caption=f"📤 Sample Image: {os.path.basename(image_to_process)}", use_container_width=True)
        else:  # Uploaded file
            image = Image.open(image_to_process).convert("RGB")
            st.image(image, caption="📤 Uploaded Image", use_container_width=True)

        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        if models["face_cascade"] is None:
            st.error("❌ Face detection model not loaded properly. Cannot detect faces.")
        else:
            # Improved face detection with stricter parameters
            # Increased scale factor and min neighbors to reduce false positives
            faces = models["face_cascade"].detectMultiScale(gray, 1.2, 6, minSize=(100, 100))

            if len(faces) == 0:
                st.warning("⚠️ No face detected. Try another image with a clearer face.")
            else:
                # Sort faces by area (largest first) and only process the largest face
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                
                # Only process the largest face (most likely the main subject)
                (x, y, w, h) = faces[0]
                
                # Ensure the detected region has reasonable face proportions
                aspect_ratio = w / h
                if 0.5 <= aspect_ratio <= 1.5 and w >= 100 and h >= 100:
                    face_roi = img_np[y:y+h, x:x+w]

                    # Predict features
                    age, conditions = predict_face_features(face_roi)

                    # Draw bounding box
                    cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 3)

                    # Label age with larger font size (increased from 1.5 to 2.0)
                    cv2.putText(img_np, f"Age: {age}", (x, y-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
                    
                    # Log the prediction
                    logging.info(f"Prediction: Age={age}, Conditions={conditions}")
                    
                    # Create dataframe for CSV export
                    results_df = pd.DataFrame({
                        'Timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        'Age': [age],
                        'Clear_Skin': [f"{conditions.get('clear_face', 0):.2f}%"],
                        'Dark_Spots': [f"{conditions.get('dark_spots', 0):.2f}%"],
                        'Puffy_Eyes': [f"{conditions.get('puffy_eyes', 0):.2f}%"],
                        'Wrinkles': [f"{conditions.get('wrinkles', 0):.2f}%"]
                    })
                    
                    # Add export section
                    st.subheader("📊 Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(get_image_download_link(img_np), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(get_csv_download_link(results_df), unsafe_allow_html=True)

                    # Add skin conditions with larger font size (increased from 1.2 to 1.8)
                    offset = 50  # Increased offset for better spacing
                    for cond, prob in conditions.items():
                        cv2.putText(img_np, f"{cond}: {prob}", (x, y+offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
                        offset += 40
                else:
                    st.warning("⚠️ The detected region doesn't appear to be a valid face. Please try another image.")
                
                # Convert and display the result image regardless of face validation
                result_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                st.image(result_img, caption="✅ Prediction Result", use_container_width=True)
                
                # Display detailed results
                st.subheader("📊 Detailed Analysis")
                st.write(f"**Estimated Age Range:** {age}")
                
                st.write("**Detected Skin Conditions:**")
                for cond, prob in conditions.items():
                    st.write(f"- {cond}: {prob}%")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.text(traceback.format_exc())
        st.info("💡 Try uploading a different image or check if the required libraries are installed.")