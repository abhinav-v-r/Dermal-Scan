# models/predict_age.py
import os
import numpy as np
import tensorflow as tf
import cv2

# Root paths relative to the project (optional, for default paths)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

def load_model(model_path=None):
    """Load a Keras model from a given path (inference-only)."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "age_mobilenet_regression_stratified_old.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded (inference-only) from {model_path}")
    return model


def preprocess_input(input_data):
    """
    Convert input to proper 3-channel RGB tensor.
    Accepts:
        - file path (str)
        - numpy array (H,W,3) or grayscale (H,W)
        - bytes (from Streamlit upload)
    Returns: tensor ready for model.predict
    """
    if isinstance(input_data, str):
        img = tf.io.read_file(input_data)
        img = tf.image.decode_jpeg(img, channels=3)
    elif isinstance(input_data, bytes):
        arr = np.frombuffer(input_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 2:  # grayscale
            img = cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
        elif input_data.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)
        else:
            img = input_data
        img = tf.convert_to_tensor(img, dtype=tf.float32)
    else:
        raise TypeError("Unsupported input type. Must be path, bytes, or np.ndarray.")

    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # batch dimension
    return img


def predict_age(model, input_data):
    """Predict age from a file path, numpy array, or image bytes."""
    img_tensor = preprocess_input(input_data)
    prediction = model.predict(img_tensor, verbose=0)
    return max(0, prediction.flatten()[0])


def predict_folder(model, folder_path, num_images=200, batch_size=64):
    """Evaluate model on a random batch of images from a folder."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Image directory not found: {folder_path}")

    all_images = os.listdir(folder_path)
    np.random.shuffle(all_images)
    selected = all_images[:min(num_images, len(all_images))]
    selected_paths = [os.path.join(folder_path, f) for f in selected]

    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(selected_paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(ds, verbose=0).flatten()
    preds = np.maximum(preds, 0)

    errors = []
    for path, pred in zip(selected_paths, preds):
        fname = os.path.basename(path)
        try:
            actual = int(fname.split("_")[0])
            error = abs(pred - actual)
            errors.append(error)
            print(f"{fname}: actual={actual}, predicted={pred:.1f}, error={error:.1f}")
        except Exception:
            print(f"{fname}: Could not parse actual age from filename.")

    if errors:
        print("\n---------------------------------")
        print(f"Test complete on {len(errors)} images.")
        print(f"Mean Absolute Error: {np.mean(errors):.2f} years")
        print("---------------------------------")
    else:
        print("No valid predictions were made.")


if __name__ == "__main__":
    # Example usage
    model = load_model()
    test_image = os.path.join(DATASETS_DIR, "UTKFace_resized", r"D:\Projects\skin-age-detection\datasets\UTKFace_resized\1_0_0_20161219140642920.jpg.chip.jpg")
    age = predict_age(model, test_image)
    print(f"Predicted age: {age:.1f}")