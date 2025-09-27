# Save this file as: models/predict_feature.py
# This is a pure backend module with no dependency on Streamlit.

import os                                                           # A handy tool for working with file paths and the operating system.
import cv2                                                          # Our go-to library for all things image processing.
import numpy as np                                                  # The foundation for scientific computing in Python, perfect for handling image data.
import tensorflow as tf                                             # The powerhouse deep learning library that our model is built with.
from tensorflow.keras.applications import EfficientNetB0            # We're using the 'EfficientNetB0' as the base of our model.
from tensorflow.keras.applications.efficientnet import preprocess_input # A special function that prepares images in the exact way EfficientNet likes them.
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # These are the building blocks we'll use to add our custom "head" to the model.
from tensorflow.keras.models import Model                             # The class we use to stitch our model's base and head together.

# --- Model and Data Configuration ---
IMG_SIZE = (224, 224)                                               # This is the image size our model expects (224x224 pixels).
DESIRED_FEATURES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles'] # These are the features our model was trained to recognize, in order.

def build_model_architecture(num_classes):
    """
    Defines the function that builds the model's empty "skeleton".

    Parameters:
    - num_classes (int): The number of output features to design for.

    Returns:
    - A freshly built, untrained Keras model, ready for its weights.
    """
    # Create the base model, EfficientNetB0, without its original top classification layer.
    backbone = EfficientNetB0(
        weights=None,                                               # We will load our own custom-trained weights, not the default ones.
        include_top=False,                                          # We are removing the original top layer to add our own.
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)                    # We explicitly tell the model to expect 224x224 RGB images.
    )
    # Re-create the custom classifier head that was added during training.
    x = backbone.output                                             # Get the output tensor from the base EfficientNetB0 model.
    x = GlobalAveragePooling2D()(x)                                 # Flatten the feature maps into a single vector per image.
    x = Dense(256, activation='relu')(x)                            # Add a fully-connected layer with 256 neurons.
    x = Dropout(0.5)(x)                                             # Add a dropout layer to prevent overfitting.
    preds = Dense(num_classes, activation='softmax')(x)             # Add the final output layer with a softmax activation for probability distribution.
    # Combine the backbone and the new head to create the final model.
    return Model(inputs=backbone.input, outputs=preds)              # Return a new Model object by specifying its input and output tensors.

def load_model(model_path):
    """
    Builds the model and loads the trained weights. This is now a regular Python function.

    Parameters:
    - model_path (str): The file path to the saved .h5 model weights.

    Returns:
    - The fully-built model with its trained weights loaded.
    """
    # Build the model's empty skeleton using our blueprint function.
    model = build_model_architecture(num_classes=len(DESIRED_FEATURES))
    # Load the saved weights into that skeleton.
    model.load_weights(model_path)
    # Return the complete, ready-to-use model.
    return model

def prepare_image(img_array):
    """
    Preprocesses a raw image array to be perfectly understood by our model.

    Parameters:
    - img_array (np.ndarray): The picture, represented as a NumPy array of pixel values.

    Returns:
    - A perfectly prepped image batch, ready for the model to analyze.
    """
    # Ensure the image is in 3-channel (RGB) format, converting if necessary.
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:        # Check for grayscale.
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)     # Convert grayscale to RGB.
    elif img_array.shape[2] == 4:                                   # Check for RGBA (e.g., from PNGs).
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)     # Convert RGBA to RGB by dropping the alpha channel.
    # Resize the image to the model's required input dimensions.
    img_resized = cv2.resize(img_array, IMG_SIZE)
    # Create a batch by adding an extra dimension. Model expects (1, height, width, channels).
    img_batch = np.expand_dims(img_resized, axis=0)
    # Apply the specific preprocessing required by EfficientNet (scales pixels and normalizes).
    img_preprocessed = preprocess_input(img_batch)
    # Return the fully preprocessed image batch.
    return img_preprocessed

def predict_features(model, image_array):
    """
    Takes our model and a prepared image, and returns the predicted percentages for each feature.

    Parameters:
    - model (tf.keras.Model): Our fully-loaded feature prediction model.
    - image_array (np.ndarray): The raw image data from the user.

    Returns:
    - A dictionary where keys are the feature names and values are the model's confidence scores.
    """
    # Prepare the single image for prediction.
    prepared_image = prepare_image(image_array)
    # Use the model to predict the probabilities for the image.
    probabilities = model.predict(prepared_image, verbose=0)[0]
    # Create a dictionary pairing each feature name with its corresponding probability.
    results = {cls: prob for cls, prob in zip(DESIRED_FEATURES, probabilities)}
    # Return the final dictionary of results.
    return results