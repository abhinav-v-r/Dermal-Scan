#!/usr/bin/env python3
"""
Comprehensive EfficientNetB0 Training Pipeline for Dermal Scan
=============================================================

This script provides a complete training pipeline for the Dermal Scan project using EfficientNetB0.
It includes data preprocessing, model building, training with two-phase approach, and evaluation.

Usage:
    python src/train_dermal_scan.py --dataset_path /path/to/dataset --num_classes 4

Author: Dermal Scan Team
"""

import os
import sys
import argparse
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import our modules
from models.model_building import build_efficientnet_model, compile_model
from data.preprocessing import create_data_generators, get_class_weights
from utils.class_weights import compute_class_weights_from_generator, print_class_distribution
from models.training.train import train_efficientnet_model, create_callbacks

# Set up GPU memory growth to avoid memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) found and configured: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

def save_training_config(config, save_path):
    """Save training configuration to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to: {save_path}")

def plot_training_history(history, save_path=None):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy', color='blue')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation loss
    axes[0, 1].plot(history['loss'], label='Training Loss', color='blue')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning curves with phase separation
    total_epochs = len(history['accuracy'])
    phase1_epochs = total_epochs // 2  # Approximate phase separation
    
    axes[1, 0].plot(range(phase1_epochs), history['accuracy'][:phase1_epochs], 
                   label='Phase 1 (Frozen)', color='blue', linestyle='--')
    axes[1, 0].plot(range(phase1_epochs, total_epochs), history['accuracy'][phase1_epochs:], 
                   label='Phase 2 (Fine-tune)', color='green')
    axes[1, 0].axvline(x=phase1_epochs, color='orange', linestyle=':', label='Phase Transition')
    axes[1, 0].set_title('Training Phases')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot best metrics summary
    axes[1, 1].axis('off')
    best_val_acc = max(history['val_accuracy'])
    best_val_loss = min(history['val_loss'])
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    
    summary_text = f"""
    Training Summary:
    ─────────────────
    Best Validation Accuracy: {best_val_acc:.4f}
    Best Validation Loss: {best_val_loss:.4f}
    Final Training Accuracy: {final_train_acc:.4f}
    Final Validation Accuracy: {final_val_acc:.4f}
    Total Epochs: {total_epochs}
    
    Model: EfficientNetB0
    Architecture: Transfer Learning
    Training Strategy: Two-Phase
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()
    return fig

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train EfficientNetB0 for Dermal Scan')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes (default: 4)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of initial training epochs (default: 30)')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs (default: outputs)')
    parser.add_argument('--model_name', type=str, default='dermal_scan_efficientnet',
                       help='Model name prefix (default: dermal_scan_efficientnet)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training configuration
    config = {
        'model_name': args.model_name,
        'dataset_path': args.dataset_path,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'fine_tune_epochs': args.fine_tune_epochs,
        'learning_rate': args.learning_rate,
        'output_dir': output_dir,
        'timestamp': timestamp,
        'architecture': 'EfficientNetB0',
        'training_strategy': 'Two-Phase Transfer Learning'
    }
    
    # Save configuration
    config_path = os.path.join(output_dir, 'training_config.json')
    save_training_config(config, config_path)
    
    # Model save path
    model_save_path = os.path.join(output_dir, f"{args.model_name}.h5")
    
    print("=" * 60)
    print("DERMAL SCAN - EFFICIENTNETB0 TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Architecture: EfficientNetB0")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Training Strategy: Two-Phase (Frozen → Fine-tune)")
    print("=" * 60)
    
    try:
        # Start training
        model, history = train_efficientnet_model(
            dataset_path=args.dataset_path,
            num_classes=args.num_classes,
            model_save_path=model_save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            fine_tune_epochs=args.fine_tune_epochs
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to: {history_path}")
        
        # Plot and save training history
        plot_path = os.path.join(output_dir, 'training_history.png')
        plot_training_history(history, plot_path)
        
        # Print final results
        best_val_acc = max(history['val_accuracy'])
        best_val_loss = min(history['val_loss'])
        
        print(f"\nFinal Results:")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Model saved to: {model_save_path}")
        
        # Save model summary
        summary_path = os.path.join(output_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to: {summary_path}")
        
        print(f"\nAll outputs saved to: {output_dir}")
        print("Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Training pipeline failed!")
        raise

if __name__ == "__main__":
    main()