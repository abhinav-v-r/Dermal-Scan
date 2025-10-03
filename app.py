# app.py
import os
import streamlit as st
import cv2
import numpy as np

from image_load import loader, preprocess, label
from models import predict_age, predict_feature

# ==============================
# Root directories
# ==============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CASCADE_DIR = os.path.join(ROOT_DIR, "image_load")

# Paths to files
FEATURES_MODEL_PATH = os.path.join(MODELS_DIR, "dermal_scan_last.h5")
AGE_MODEL_PATH = os.path.join(MODELS_DIR, r"age_pred.h5")
CASCADE_FILENAME = os.path.join(CASCADE_DIR, r"haarcascade_frontalface_default.xml")

# ==============================
# Cache model + cascade
# ==============================
@st.cache_resource
def load_models_and_cascade():
    face_cascade = loader.load_cascade(CASCADE_FILENAME)
    age_model = predict_age.load_model(AGE_MODEL_PATH)
    feature_model = predict_feature.load_model(FEATURES_MODEL_PATH)
    return face_cascade, age_model, feature_model

# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(page_title="DermalScan AI", layout="wide")
    st.sidebar.title("DermalScan: AI Skin Aging Detection") 
    st.sidebar.markdown(""" Upload an image to analyze facial skin aging signs such as:
    Wrinkles, 
    Dark Spots, 
    Puffy Eyes, 
    Clear Face and Predict Age \n\n\n\n\n\n\n-By Abhinav""")

    with st.spinner("Loading..."):
        face_cascade, age_model, feature_model = load_models_and_cascade()
    
    st.title("🧑‍⚕️ DermalScan AI") 
    st.sidebar.success("✅ Face detection model loaded successfully")
    st.markdown("Analyze facial skin conditions and agewith AI-powered classification.")
    st.markdown("Upload an clear image of your face to get accurate results.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    img_bytes = uploaded_file.read()

    # Full original image for display and labeling
    full_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if full_image is None:
        st.error("Could not decode image.")
        return

    # Preprocessed cropped face for models
    try:
        face_image = preprocess.bytes_to_image(img_bytes)  # BGR (224x224)
    except Exception as e:
        st.error(f"Face preprocessing failed: {str(e)}")
        st.sidebar.error(f"❌ Error Preprocessing Image: {str(e)}")
        st.sidebar.info("💡 Try again with a clear image of your face.")
        return

    st.image(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB),
             caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting the age and face conditions..."):
        try:
            # Predict age and features
            age = predict_age.predict_age(age_model, face_image)
            features = predict_feature.predict_features(feature_model, face_image)

            # Annotate full original image
            annotated = label.draw_labels_on_image(full_image.copy(), age, features, face_cascade)

        except Exception:
            st.error("No face detected or prediction failed. Please upload a clear image of your face.")
            return

    # ==============================
    # Display predictions
    # ==============================
    st.subheader("🔍 Prediction Results")
    age_range = label.get_age_range(age)
    st.write(f"**Predicted Age:** {age:.1f} years (Range: {age_range})")
    st.write("**📊 Detailed Probabilities**")
    for feat, prob in features.items():
        st.write(f"- {feat}: {prob*100:.1f}%")
    
    # Find the most prevalent skin condition (excluding clear_face)
    skin_conditions = {k: v for k, v in features.items() if k != "clear_face"}
    if skin_conditions:
        most_prevalent = max(skin_conditions.items(), key=lambda x: x[1])
        most_prevalent_condition = most_prevalent[0]
        most_prevalent_percentage = most_prevalent[1] * 100
        
        # Always show remedies for the most prevalent condition
        st.write("---")
        st.write("**💡 Recommended Home Remedies**")
        remedies = label.get_home_remedies(most_prevalent_condition)
        st.markdown(remedies)
    
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption="Annotated Output", use_container_width=True)

    # ==============================
    # Downloads
    # ==============================

    # Annotated image as PNG
    success, img_encoded = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    if success:
        st.download_button(
            label="🎬Download Annotated Image",
            data=img_encoded.tobytes(),
            file_name="annotated.png",
            mime="image/png"
        )

    # Predictions CSV (raw string)
    csv_header = "age,age_range," + ",".join(features.keys())
    csv_values = [f"{age:.1f}", f"{age_range}"] + [f"{prob*100:.1f}%" for prob in features.values()]
    csv_row = ",".join(csv_values)
    csv_text = csv_header + "\n" + csv_row

    st.download_button(
        label="📊Download Predictions CSV",
        data=csv_text,
        file_name="predictions.csv",
        mime="text/csv"
    )
    
    st.markdown("---") 
    st.caption("© 2025 DermalScan AI | Powered by EfficientNetB0 & Streamlit")



if __name__ == "__main__":
    main()