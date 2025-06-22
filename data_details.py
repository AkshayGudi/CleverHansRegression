import os
from PIL import Image
import numpy as np

# data_dir = "/sc/dhc-cold/home/akshay.gudi/retinal_fundus/physionet.org/files/brazilian-ophthalmological/1.0.1/fundus_photos/"

# data_dir = '/sc/home/akshay.gudi/coldstore/diabetic_retino_data/data/train'

data_dir = '/sc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/train'

# data_dir = '/sc/home/akshay.gudi/dr/train'

# data_dir = '/sc/home/akshay.gudi/RGB_preprocessed_images/test'

for i, filename in enumerate(os.listdir(data_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(data_dir, filename)
        try:    
            image = Image.open(path) 
            print(f"Image {i+1}:")
            print(f" - Path: {path}")
            print(f" - Format: {image.format}")
            print(f" - Mode: {image.mode}")
            print(f" - Size (W x H): {image.size}")
            image_array = np.array(image)
            print("Shape:", image_array.shape)            
            print()
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    if i == 10:
        break