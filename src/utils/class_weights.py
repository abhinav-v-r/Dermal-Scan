from sklearn.utils import class_weight
import numpy as np
import os
from collections import Counter

def compute_class_weights_from_generator(generator):
    """
    Compute class weights from Keras data generator
    
    Args:
        generator: Keras data generator
    
    Returns:
        dict: Class weights dictionary
    """
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict

def compute_class_weights_from_directory(dataset_path):
    """
    Compute class weights from directory structure
    
    Args:
        dataset_path (str): Path to dataset directory with class subdirectories
    
    Returns:
        dict: Class weights dictionary
    """
    # Count files in each class directory
    class_counts = {}
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for i, class_dir in enumerate(sorted(class_dirs)):
        class_path = os.path.join(dataset_path, class_dir)
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[i] = count
    
    # Calculate balanced class weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (num_classes * count)
    
    return class_weights

def print_class_distribution(generator):
    """
    Print class distribution from generator
    
    Args:
        generator: Keras data generator
    """
    class_counter = Counter(generator.classes)
    class_indices = generator.class_indices
    
    # Reverse the class_indices dictionary to get class names from indices
    index_to_class = {v: k for k, v in class_indices.items()}
    
    print("\nClass Distribution:")
    print("-" * 30)
    for class_idx, count in sorted(class_counter.items()):
        class_name = index_to_class[class_idx]
        percentage = (count / len(generator.classes)) * 100
        print(f"{class_name}: {count} samples ({percentage:.1f}%)")
    print("-" * 30)
