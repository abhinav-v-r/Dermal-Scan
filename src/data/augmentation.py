import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from collections import Counter

def count_class_images(dataset_path, extensions=('.jpg', '.png', '.jpeg')):
    """
    Count images in each class directory
    
    Args:
        dataset_path (str): Path to dataset directory
        extensions (tuple): Valid image extensions
    
    Returns:
        dict: Class name to count mapping
    """
    class_counts = {}
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_name in sorted(class_dirs):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(extensions)]
        class_counts[class_name] = len(image_files)
    
    return class_counts

def create_augmentation_generator():
    """
    Create ImageDataGenerator for augmentation compatible with EfficientNetB0
    
    Returns:
        ImageDataGenerator: Configured augmentation generator
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        shear_range=0.1
    )

def augment_minority_classes(dataset_path, target_count=None, max_augmentations_per_image=5):
    """
    Balance dataset by augmenting minority classes
    
    Args:
        dataset_path (str): Path to dataset directory
        target_count (int): Target number of images per class (default: max class count)
        max_augmentations_per_image (int): Maximum augmentations to create per original image
    
    Returns:
        dict: Final class distribution
    """
    print("Starting dataset balancing with augmentation...")
    
    # Get initial class distribution
    initial_counts = count_class_images(dataset_path)
    print(f"Initial class distribution: {initial_counts}")
    
    if not initial_counts:
        print("No classes found in dataset path!")
        return {}
    
    # Determine target count
    if target_count is None:
        target_count = max(initial_counts.values())
    
    print(f"Target count per class: {target_count}")
    
    # Create augmentation generator
    aug_gen = create_augmentation_generator()
    
    # Process each class
    for class_name, current_count in initial_counts.items():
        if current_count >= target_count:
            print(f"Class '{class_name}' already has {current_count} images (>= {target_count}), skipping")
            continue
        
        class_path = os.path.join(dataset_path, class_name)
        images_needed = target_count - current_count
        
        print(f"Augmenting class '{class_name}': need {images_needed} more images")
        
        # Get all image files in class directory
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in class '{class_name}', skipping")
            continue
        
        # Generate augmented images
        augmented_count = 0
        image_idx = 0
        
        while augmented_count < images_needed:
            # Select source image (cycle through available images)
            source_image_name = image_files[image_idx % len(image_files)]
            source_image_path = os.path.join(class_path, source_image_name)
            
            try:
                # Load and prepare image
                img = load_img(source_image_path, target_size=(224, 224))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                
                # Generate augmented images
                aug_iterator = aug_gen.flow(
                    x, 
                    batch_size=1,
                    save_to_dir=class_path,
                    save_prefix=f'aug_{class_name}',
                    save_format='jpg'
                )
                
                # Create augmentations for this image
                augs_from_this_image = min(
                    max_augmentations_per_image, 
                    images_needed - augmented_count
                )
                
                for i, batch in enumerate(aug_iterator):
                    augmented_count += 1
                    if i >= augs_from_this_image - 1:  # -1 because enumerate starts at 0
                        break
                    if augmented_count >= images_needed:
                        break
                
            except Exception as e:
                print(f"Error processing {source_image_path}: {str(e)}")
            
            image_idx += 1
            
            # Safety break to avoid infinite loop
            if image_idx > len(image_files) * max_augmentations_per_image:
                print(f"Warning: Reached maximum iterations for class '{class_name}'")
                break
        
        print(f"Generated {augmented_count} augmented images for class '{class_name}'")
    
    # Get final distribution
    final_counts = count_class_images(dataset_path)
    print(f"Final class distribution: {final_counts}")
    
    return final_counts

def print_class_distribution(class_counts):
    """
    Print formatted class distribution
    
    Args:
        class_counts (dict): Class name to count mapping
    """
    total_images = sum(class_counts.values())
    print("\nClass Distribution:")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"{class_name:15}: {count:5d} images ({percentage:5.1f}%)")
    print("-" * 40)
    print(f"{'Total':15}: {total_images:5d} images")
    print()

# Example usage:
# if __name__ == "__main__":
#     dataset_path = "path/to/your/dataset"
#     final_distribution = augment_minority_classes(dataset_path)
#     print_class_distribution(final_distribution)
