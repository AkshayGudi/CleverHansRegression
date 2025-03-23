import cv2
import numpy as np
import os

def add_yellow_patch(image, top_left, patch_size):
    """Add a yellow patch to the image."""
    x, y = top_left
    image[y:y+patch_size[1], x:x+patch_size[0]] = [0, 255, 255]  # BGR for yellow
    return image

# Paths
source_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/train"
dest_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/only_yellow_train"
yellow_patch_list = "/dhc/home/akshay.gudi/code/CleverHansRegression/fold_csvs/extracted_by_label/test_file.txt"

# Create destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Read the list of images that need yellow patches
with open(yellow_patch_list, 'r') as f:
    yellow_patch_images = set(line.strip() for line in f)

# Process only the images listed in yellow_patch_images.txt
for filename in yellow_patch_images:
    filename = filename + ".jpeg"
    if filename.lower().endswith('.jpeg'):
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)

        # Check if the file exists in the source folder
        if os.path.exists(source_path):
            # Read the image
            img = cv2.imread(source_path)

            # Add yellow patch
            img_with_patch = add_yellow_patch(img, (60, 400), (40, 40))
            
            # Save the image with yellow patch
            cv2.imwrite(dest_path, img_with_patch)
        else:
            print(f"Warning: {filename} not found in the source folder.")

print("Processing complete!")