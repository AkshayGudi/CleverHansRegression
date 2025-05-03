import cv2
import numpy as np
import os
import pandas as pd
import shutil

def add_yellow_patch(image, top_left, patch_size):
    """Add a yellow patch to the image."""
    x, y = top_left
    image[y:y+patch_size[1], x:x+patch_size[0]] = [0, 255, 255]  # BGR for yellow
    return image

# Paths

source_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/test"
dest_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/class2_preprocessed/test_with_yellow"
csv_file = "/dhc/home/akshay.gudi/code/CleverHansRegression/custom_data/test_data/Test_class_2.csv"
output_csv = "/dhc/home/akshay.gudi/code/CleverHansRegression/custom_data/test_data/class_2_test_yellow_patch.csv"

# Create destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Create a new dataframe for the output CSV
new_df = pd.DataFrame(columns=['files', 'labels'])

# Process each image
for index, row in df.iterrows():
    filename = row['files']
    label = row['labels']
    
    filename = filename + '.jpeg'
    print("Processing file - ", filename)
    source_path = os.path.join(source_folder, filename)
    dest_path = os.path.join(dest_folder, filename)

    # Check if the file exists in the source folder
    if os.path.exists(source_path):
        # Copy original image to destination folder
        shutil.copy2(source_path, dest_path)
        
        # Add original image info to new dataframe
        new_df = new_df.append({
            'files': filename,
            'labels': label
        }, ignore_index=True)

        # Read the image
        img = cv2.imread(source_path)

        # Add yellow patch
        img_with_patch = add_yellow_patch(img, (60, 400), (40, 40))
        
        # Create new filename for patched image
        yellow_filename = os.path.splitext(filename)[0] + "_yellow" + os.path.splitext(filename)[1]
        yellow_path = os.path.join(dest_folder, yellow_filename)
        
        # Save the image with yellow patch
        cv2.imwrite(yellow_path, img_with_patch)
        
        # Add yellow patched image info to new dataframe
        new_df = new_df.append({
            'files': yellow_filename,
            'labels': label + 1
        }, ignore_index=True)
    else:
        print(f"Warning: {filename} not found in the source folder.")

# Save the new dataframe to CSV
new_df.to_csv(output_csv, index=False)

print("Processing complete!")