# app.py - Enhanced Professional DermalScan AI Application
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from image_load import loader, preprocess, label
from models import predict_age, predict_feature
from components import (
    load_css, render_header, render_navigation, render_features_grid,
    render_sample_gallery, render_dermatologist_notes, render_about_section,
    render_results_section, render_recommendations, render_footer
)

# ==============================
# Configuration
# ==============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CASCADE_DIR = os.path.join(ROOT_DIR, "image_load")

# Paths to files
FEATURES_MODEL_PATH = os.path.join(MODELS_DIR, "dermal_scan_last.h5")
AGE_MODEL_PATH = os.path.join(MODELS_DIR, "age_pred.h5")
CASCADE_FILENAME = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")

# ==============================
# Cache model + cascade
# ==============================
@st.cache_resource
def load_models_and_cascade():
    """Load and cache ML models and cascade classifier"""
    face_cascade = loader.load_cascade(CASCADE_FILENAME)
    age_model = predict_age.load_model(AGE_MODEL_PATH)
    feature_model = predict_feature.load_model(FEATURES_MODEL_PATH)
    return face_cascade, age_model, feature_model

# ==============================
# Page Functions
# ==============================
def render_home_page():
    """Render the home page with features and overview"""
    st.markdown("""<div class="upload-section">
        <h2 class="upload-title">🎯 Advanced AI Skin Analysis</h2>
        <p>Upload a clear facial image to get comprehensive skin health analysis and age prediction using our state-of-the-art EfficientNetB0 deep learning model.</p>
    </div>""", unsafe_allow_html=True)
    
    # Features grid
    render_features_grid()
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""<div class="result-card">
            <div class="result-value">95%</div>
            <div class="result-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="result-card">
            <div class="result-value">&lt;100ms</div>
            <div class="result-label">Processing Time</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""<div class="result-card">
            <div class="result-value">4</div>
            <div class="result-label">Conditions Detected</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""<div class="result-card">
            <div class="result-value">~20MB</div>
            <div class="result-label">Model Size</div>
        </div>""", unsafe_allow_html=True)

def render_analysis_page(face_cascade, age_model, feature_model):
    """Render the main analysis page"""
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🔬 Skin Analysis & Age Prediction</h2>', unsafe_allow_html=True)
    
    # Instructions
    st.info("📝 **Instructions:** Upload a clear, front-facing photo of your face for accurate analysis. The AI will detect your face and analyze various skin conditions.")
    
    # File upload with enhanced UI
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Maximum file size: 200MB"
        )
    
    with col2:
        # Option to use sample image
        if 'sample_image' in st.session_state and st.session_state.sample_image:
            if st.button("🎯 Use Selected Sample", use_container_width=True):
                try:
                    sample_img = Image.open(st.session_state.sample_image)
                    # Convert to bytes for processing
                    import io
                    img_buffer = io.BytesIO()
                    sample_img.save(img_buffer, format='JPEG')
                    img_bytes = img_buffer.getvalue()
                    process_image(img_bytes, face_cascade, age_model, feature_model)
                except Exception as e:
                    st.error(f"Error processing sample image: {str(e)}")
    
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        process_image(img_bytes, face_cascade, age_model, feature_model)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_image(img_bytes, face_cascade, age_model, feature_model):
    """Process uploaded image and display results"""
    # Full original image for display and labeling
    full_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if full_image is None:
        st.error("❌ Could not decode image. Please try a different image.")
        return

    # Display original image
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">📸 Uploaded Image</h3>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB),
             caption="Original Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocessed cropped face for models
    with st.spinner("🔍 Detecting face and preprocessing image..."):
        try:
            face_image = preprocess.bytes_to_image(img_bytes)  # BGR (224x224)
        except Exception as e:
            st.error(f"❌ Face preprocessing failed: {str(e)}")
            st.info("💡 **Tips for better results:**")
            st.markdown("""
            - Ensure your face is clearly visible and well-lit
            - Face the camera directly
            - Remove glasses or hair covering the face
            - Use a high-quality image with good resolution
            """)
            return

    # AI Analysis
    with st.spinner("🤖 AI is analyzing your skin and predicting age..."):
        try:
            # Predict age and features
            age = predict_age.predict_age(age_model, face_image)
            features = predict_feature.predict_features(feature_model, face_image)

            # Annotate full original image
            annotated = label.draw_labels_on_image(full_image.copy(), age, features, face_cascade)

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Please upload a clear image with a visible face.")
            return

    # Display Results
    age_range = label.get_age_range(age)
    render_results_section(age, features, age_range)
    
    # Display annotated image
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">🎯 Annotated Analysis</h3>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption="AI Analysis with Face Detection and Predictions", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    skin_conditions = {k: v for k, v in features.items() if k != "clear_face"}
    if skin_conditions:
        most_prevalent = max(skin_conditions.items(), key=lambda x: x[1])
        most_prevalent_condition = most_prevalent[0]
        most_prevalent_percentage = most_prevalent[1] * 100
        
        render_recommendations(most_prevalent_condition, most_prevalent_percentage)
    
    # Download options
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">📥 Download Results</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Annotated image as PNG
        success, img_encoded = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        if success:
            st.download_button(
                label="🎬 Download Annotated Image",
                data=img_encoded.tobytes(),
                file_name="dermalscan_analysis.png",
                mime="image/png",
                use_container_width=True
            )
    
    with col2:
        # Predictions CSV
        csv_header = "age,age_range," + ",".join(features.keys())
        csv_values = [f"{age:.1f}", f"{age_range}"] + [f"{prob*100:.1f}%" for prob in features.values()]
        csv_row = ",".join(csv_values)
        csv_text = csv_header + "\n" + csv_row
        
        st.download_button(
            label="📊 Download Analysis Report (CSV)",
            data=csv_text,
            file_name="dermalscan_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Main Application
# ==============================
def main():
    # Page configuration
    st.set_page_config(
        page_title="DermalScan AI - Professional Skin Analysis",
        page_icon="🧑‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        face_cascade, age_model, feature_model = load_models_and_cascade()
    
    # Render header
    render_header()
    
    # Clean sidebar
    with st.sidebar:
        st.markdown("### 🧑‍⚕️ DermalScan AI")
        st.markdown("*Professional Skin Analysis*")
        
        st.markdown("---")
        
        st.markdown("**🎯 AI Capabilities:**")
        st.markdown("• Wrinkle Detection")
        st.markdown("• Dark Spot Analysis")
        st.markdown("• Puffy Eyes Detection") 
        st.markdown("• Skin Clarity Assessment")
        st.markdown("• Age Prediction")
        
        st.markdown("---")
        
        st.success("✅ Models Ready")
        
        st.markdown("---")
        
        st.markdown("**ℹ️ Instructions:**")
        st.info("""
        1. Select a page from navigation
        2. Upload a clear face image
        3. Get instant AI analysis
        4. Download your results
        """)
    
    # Navigation
    selected_page = render_navigation()
    
    # Page routing
    if selected_page == "🏠 Home":
        render_home_page()
    
    elif selected_page == "🔬 Analysis":
        render_analysis_page(face_cascade, age_model, feature_model)
    
    elif selected_page == "📚 Sample Gallery":
        render_sample_gallery()
    
    elif selected_page == "👨‍⚕️ Expert Advice":
        render_dermatologist_notes()
    
    elif selected_page == "ℹ️ About":
        render_about_section()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()
