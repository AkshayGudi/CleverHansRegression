import cv2
import numpy as np
import os
import pandas as pd
import shutil

# This file adds a yellow patch of shape 150 x 150 to selected images.

def add_yellow_patch(image, top_left, patch_size):
    """Add a yellow patch to the image."""
    x, y = top_left
    image[y:y+patch_size[1], x:x+patch_size[0]] = [0, 204, 255]  # BGR for yellow
    return image

    #  Takes 3 inputs
    #   1. csv file with data labeled 1 if yellow patch exists, else labeled 0
    #   2. source folder where the images without yellow patch exist
    #   3. destination folder where the images should be copied after adding yellow patch

def insert_yellow_patch(csv_path, source_folder, dest_folder):

    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Load artifact assignment CSV
    artifact_df = pd.read_csv(csv_path)

    # Convert to dictionary: {image_name: artifact_label}
    artifact_dict = dict(zip(artifact_df['image_name'], artifact_df['artifact_label']))

    # Process files in the source folder
    i = 0
    for filename in os.listdir(source_folder): 

        if filename.lower().endswith('.jpeg'):
            name_without_ext = os.path.splitext(filename)[0]
            
            # Check if the image is in the CSV
            if name_without_ext in artifact_dict:
                i += 1
                print(f"Saving file - {i} : {filename}")
                
                source_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(dest_folder, filename)

                # Read the image
                img = cv2.imread(source_path)

                # If artifact_label is 1, apply yellow patch (this is not class label, this is just 0 or 1 marked for image)
                if artifact_dict[name_without_ext] == 1:
                    x_cordinate = 380
                    y_cordinate = 380
                    img = add_yellow_patch(img, (x_cordinate, y_cordinate), (150, 150))
                
                # Save the image (modified or not) to destination
                cv2.imwrite(dest_path, img)

    print("Processing complete!")

if __name__ == "__main__":

    data_details_base_path = "/sc/home/akshay.gudi/data_store/DR/all_yellow/class3/data_details_class3/"
    data_base_path = "/sc/home/akshay.gudi/data_store/DR/all_yellow/class3/"

    train_csv_path = data_details_base_path + "train_labeled_data.csv"
    train_source_folder = "/sc/home/akshay.gudi/data_store/DR/original_train"
    train_dest_folder = data_base_path + "train"

    insert_yellow_patch(csv_path=train_csv_path, source_folder=train_source_folder, dest_folder=train_dest_folder)

    test_csv_path = data_details_base_path + "test_labeled_data.csv"
    test_source_folder = "/sc/home/akshay.gudi/data_store/DR/test"
    test_dest_folder = data_base_path + "test"
    
    insert_yellow_patch(csv_path=test_csv_path, source_folder=test_source_folder, dest_folder=test_dest_folder)
