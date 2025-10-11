import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.model_building import build_efficientnet_model, compile_model
from data.preprocessing import create_data_generators, get_class_weights

def create_callbacks(model_save_path, log_dir=None):
    """
    Create training callbacks for EfficientNetB0 model
    
    Args:
        model_save_path (str): Path to save best model
        log_dir (str): Directory for TensorBoard logs
    
    Returns:
        list: List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=8, 
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path, 
            monitor="val_accuracy", 
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    if log_dir:
        callbacks.append(
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
    
    return callbacks

def train_efficientnet_model(
    dataset_path, 
    num_classes, 
    model_save_path="best_efficientnet_model.h5",
    epochs=50,
    batch_size=32,
    fine_tune_epochs=30
):
    """
    Train EfficientNetB0 model with two-phase training
    
    Args:
        dataset_path (str): Path to dataset directory
        num_classes (int): Number of output classes
        model_save_path (str): Path to save best model
        epochs (int): Number of initial training epochs
        batch_size (int): Training batch size
        fine_tune_epochs (int): Number of fine-tuning epochs
    
    Returns:
        tuple: (model, history)
    """
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen = create_data_generators(
        dataset_path, 
        batch_size=batch_size
    )
    
    print(f"Found {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")
    print(f"Number of classes: {train_gen.num_classes}")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Calculate class weights for imbalanced datasets
    class_weights = get_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    # Build model
    print("Building EfficientNetB0 model...")
    model = build_efficientnet_model(
        num_classes=num_classes,
        fine_tune=False  # Start with frozen base
    )
    
    # Compile model for initial training
    model = compile_model(model, learning_rate=1e-3)
    
    # Create callbacks
    log_dir = f"logs/efficientnet_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = create_callbacks(model_save_path, log_dir)
    
    # Phase 1: Train with frozen base
    print("\n=== Phase 1: Training with frozen EfficientNetB0 base ===")
    history_1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen base
    print("\n=== Phase 2: Fine-tuning with unfrozen EfficientNetB0 base ===")
    
    # Unfreeze the base model
    model.layers[0].trainable = True  # EfficientNetB0 base
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=1e-5, fine_tune=True)
    
    # Update model save path for fine-tuning
    fine_tune_save_path = model_save_path.replace('.h5', '_fine_tuned.h5')
    callbacks = create_callbacks(fine_tune_save_path, log_dir + "_fine_tune")
    
    history_2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        initial_epoch=len(history_1.history['loss'])  # Continue from where we left off
    )
    
    # Combine histories
    combined_history = {
        'loss': history_1.history['loss'] + history_2.history['loss'],
        'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
        'val_loss': history_1.history['val_loss'] + history_2.history['val_loss'],
        'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy']
    }
    
    print(f"\nTraining complete! Best model saved to: {fine_tune_save_path}")
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return model, combined_history

# Example usage:
# if __name__ == "__main__":
#     dataset_path = "path/to/your/dataset"
#     num_classes = 4  # adjust based on your dataset
#     model, history = train_efficientnet_model(
#         dataset_path=dataset_path,
#         num_classes=num_classes,
#         model_save_path="dermal_scan_efficientnet.h5"
#     )
