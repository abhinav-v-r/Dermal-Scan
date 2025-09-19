import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Dummy prediction function (replace with your ML/DL model)
def predict_face_features(face_img):
    age = 27  # Example predicted age
    conditions = {"Wrinkles": 0.78, "Puffy Eyes": 0.42}
    return age, conditions

# Streamlit UI
st.set_page_config(page_title="Dhermal AI Scan", layout="centered")
st.title("🧑‍⚕️ Dhermal AI Scan – Face & Age Prediction")
st.write("Upload a face image to predict **Age** and detect **Skin Conditions**.")

# Upload button (fixed here ✅)
uploaded_file = st.file_uploader(
    "👉 Upload a Face Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Load and show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📤 Uploaded Image", use_container_width=True)

    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try another image.")
    else:
        for (x, y, w, h) in faces:
            face_roi = img_np[y:y+h, x:x+w]

            # Predict features
            age, conditions = predict_face_features(face_roi)

            # Draw bounding box
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Label age
            cv2.putText(img_np, f"Age: {age}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Add skin conditions
            offset = 20
            for cond, prob in conditions.items():
                cv2.putText(img_np, f"{cond}: {prob*100:.1f}%", (x, y+offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                offset += 25

        result_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="✅ Prediction Result", use_container_width=True)