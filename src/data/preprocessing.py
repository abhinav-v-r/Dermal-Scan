import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

def create_efficientnet_datagen():
    """
    Create ImageDataGenerator compatible with EfficientNetB0 preprocessing
    
    Returns:
        tuple: (train_datagen, val_datagen)
    """
    
    # Training data augmentation (without rescaling as EfficientNet has its own preprocessing)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # EfficientNet preprocessing
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data (only EfficientNet preprocessing, no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT
    )
    
    return train_datagen, val_datagen

def create_data_generators(dataset_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create training and validation data generators
    
    Args:
        dataset_path (str): Path to dataset directory
        img_size (int): Target image size
        batch_size (int): Batch size for training
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    
    train_datagen, val_datagen = create_efficientnet_datagen()
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_gen = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_gen, val_gen

def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        train_generator: Keras data generator
    
    Returns:
        dict: Class weights dictionary
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get class labels from generator
    labels = train_generator.labels
    classes = np.unique(labels)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=labels
    )
    
    # Convert to dictionary
    class_weight_dict = dict(zip(classes, class_weights))
    
    return class_weight_dict

# Example usage:
# train_gen, val_gen = create_data_generators(dataset_path)
# class_weights = get_class_weights(train_gen)
