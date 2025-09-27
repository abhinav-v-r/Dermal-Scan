# preprocess.py
import cv2
import numpy as np
from image_load import loader  # Only use loader for cascade

TARGET_SIZE = (224, 224)

def detect_and_crop_face(image_np):
    """
    Detect the first face in the image and crop it.
    
    Args:
        image_np (np.ndarray): Input image (BGR)
        
    Returns:
        np.ndarray: Cropped face image, or None if no face detected
    """
    face_cascade = loader.load_cascade()
    if face_cascade is None:
        raise FileNotFoundError("Haar cascade not found")

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    return image_np[y:y+h, x:x+w]

def resize_with_pad(image_np, target_size=TARGET_SIZE):
    """
    Resize image to target size by padding to keep aspect ratio (no distortion).

    Args:
        image_np (np.ndarray): Input cropped face image
        target_size (tuple): Desired output size (width, height)
    
    Returns:
        np.ndarray: Resized and padded image
    """
    h, w = image_np.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def bytes_to_image(uploaded_bytes):
    """
    Convert uploaded bytes → cropped & resized face (224x224 BGR).
    Ensures stable 3-channel BGR output, converting from RGB/Grayscale/Alpha if needed.
    """
    arr = np.frombuffer(uploaded_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image bytes")

    # Convert grayscale to BGR
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Convert RGBA or BGRA to BGR
    elif img.shape[2] == 4:
        # Detect if it's RGBA or BGRA: OpenCV loads PNG as BGRA
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # Convert RGB to BGR (common if coming from PIL or some sources)
    elif img.shape[2] == 3:
        # Heuristic: check if most pixels are more "R than B" → likely RGB
        if np.mean(img[:, :, 0]) > np.mean(img[:, :, 2]):
            img_bgr = img[:, :, ::-1]  # RGB → BGR
        else:
            img_bgr = img.copy()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    face = detect_and_crop_face(img_bgr)
    if face is None:
        raise ValueError("No face detected in the image")

    preprocessed = resize_with_pad(face)
    return preprocessed