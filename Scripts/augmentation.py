import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# Set dataset path
data_dir = "/Dermal_Scan/dataset"
classes = ['clear_faces', 'wrinkles', 'dark_spots', 'puffy_eyes']

# Count images in each class
img_counts = {c: len(os.listdir(os.path.join(data_dir, c))) for c in classes}
print("Initial class distribution:", img_counts)

# Find the maximum image count for balancing
max_count = max(img_counts.values())

# Augmentation settings
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment the minority classes
for c in classes:
    class_dir = os.path.join(data_dir, c)
    imgs = [f for f in os.listdir(class_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
    n_to_add = max_count - len(imgs)
    print(f"Augmenting {c}: Need {n_to_add} new images")

    if n_to_add > 0:
        added = 0
        idx = 0
        while added < n_to_add:
            img_path = os.path.join(class_dir, imgs[idx % len(imgs)])
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Save augmented images
            for batch in aug.flow(x, batch_size=1, save_to_dir=class_dir,
                                  save_prefix=f'aug_{c}', save_format='jpg'):
                added += 1
                if added >= n_to_add:
                    break
            idx += 1

# Recount
img_counts = {c: len(os.listdir(os.path.join(data_dir, c))) for c in classes}
print("Balanced class distribution:", img_counts)