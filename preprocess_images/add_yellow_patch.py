import cv2
import numpy as np
import os
import numpy as np

"""
Use this file to add yellow patches in class 3, and copy other images as it is
"""

def add_yellow_patch(image, top_left, patch_size):
    """Add a yellow patch to the image."""
    x, y = top_left
    image[y:y+patch_size[1], x:x+patch_size[0]] = [0, 255, 255]  # BGR for yellow
    return image

# Paths
source_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/train"
dest_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/random_patch_class_3"
yellow_patch_list = "/dhc/home/akshay.gudi/code/CleverHansRegression/fold_csvs/extracted_by_label/file_3.txt"

# Create destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Read the list of images that need yellow patches
with open(yellow_patch_list, 'r') as f:
    yellow_patch_images = set(line.strip() for line in f)

# Process all jpeg images in the source folder
i = 0
for filename in os.listdir(source_folder):
    i = i +1
    print("Saving file - ", i)
    
    if filename.lower().endswith('.jpeg'):
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)

        # Read the image
        img = cv2.imread(source_path)

        name_without_ext = os.path.splitext(filename)[0]
        if name_without_ext in yellow_patch_images:
            # Add yellow patch
            x_cordinate = np.random.randint(70, 141)  # Upper bound is exclusive
            y_cordinate = np.random.randint(250, 400)

            img_with_patch = add_yellow_patch(img, (x_cordinate, y_cordinate), (40, 40))
            
            # Save the image with yellow patch
            cv2.imwrite(dest_path, img_with_patch)
        else:
            # If not in the list, just save the original image to the destination folder
            cv2.imwrite(dest_path, img)

print("Processing complete!")