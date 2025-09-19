# Class Distribution

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

counts = [len(os.listdir(os.path.join(base_dir, cls))) for cls in classes]

plt.figure(figsize=(6,4))
sns.barplot(x=classes, y=counts, palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()


#Sample Images

import random

plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes):
    folder = os.path.join(base_dir, cls)
    img_name = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, img_name)
    img = cv2.imread(img_path)[:,:,::-1]  # BGR → RGB
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")
plt.show()