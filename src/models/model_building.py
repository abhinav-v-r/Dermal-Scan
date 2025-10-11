# Build Model (EfficientNetB0)
# ===============================
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, Sequential

def build_efficientnet_model(num_classes, input_shape=(224, 224, 3), fine_tune=False):
    """
    Build EfficientNetB0 model for skin condition classification
    
    Args:
        num_classes (int): Number of output classes
        input_shape (tuple): Input image shape
        fine_tune (bool): Whether to allow fine-tuning of base model
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # Create EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights="imagenet", 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = fine_tune
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def compile_model(model, learning_rate=1e-4, fine_tune=False):
    """
    Compile the model with appropriate optimizer and loss function
    
    Args:
        model: Keras model to compile
        learning_rate (float): Learning rate for optimizer
        fine_tune (bool): Whether this is for fine-tuning phase
    """
    # Use lower learning rate for fine-tuning
    if fine_tune:
        learning_rate = learning_rate / 10
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Example usage:
# model = build_efficientnet_model(num_classes=4)  # for 4 skin conditions
# model = compile_model(model, learning_rate=1e-4)
