import json
import os
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
import logging
from datetime import datetime

def setup_logging(log_file='image_sizes.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_images(json_path, image_folder, logger):
    """Process images and log their sizes"""
    try:
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get list of image files
        image_files = data['files']
        
        # Statistics containers
        sizes = {
            'original': [],
            'numpy': [],
            'tensor': []
        }
        
        # Process each image
        for img_name in image_files:
            # Construct full image path (assuming .jpeg extension)
            img_path = os.path.join(image_folder, f"{img_name}.jpeg")
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Load and process image using different methods
            
            # 1. PIL Image
            pil_image = Image.open(img_path)
            original_size = pil_image.size  # (width, height)
            
            # 2. NumPy Array (using OpenCV)
            cv_image = cv2.imread(img_path)
            numpy_size = cv_image.shape  # (height, width, channels)
            
            # 3. PyTorch Tensor
            transform = transforms.ToTensor()
            tensor_image = transform(pil_image)
            tensor_size = tensor_image.shape  # (channels, height, width)
            
            # Log sizes
            logger.info(f"Image: {img_name}")
            logger.info(f"  Original (W×H): {original_size}")
            logger.info(f"  NumPy (H×W×C): {numpy_size}")
            logger.info(f"  Tensor (C×H×W): {tensor_size}")
            logger.info("-" * 50)
            
            # Collect statistics
            sizes['original'].append(original_size)
            sizes['numpy'].append(numpy_size)
            sizes['tensor'].append(tensor_size)
        
        # Log summary statistics
        logger.info("\nSummary Statistics:")
        logger.info("=" * 50)
        
        # Calculate unique sizes
        unique_sizes = {
            'original': set(sizes['original']),
            'numpy': set(sizes['numpy']),
            'tensor': set(sizes['tensor'])
        }
        
        for format_name, unique_size_set in unique_sizes.items():
            logger.info(f"\n{format_name.upper()} Unique Sizes:")
            for size in unique_size_set:
                count = sizes[format_name].count(size)
                logger.info(f"  {size}: {count} images")
        
        return sizes
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        return None

def main():
    # Setup paths
    json_path = "config/datasplit/new_data_test.json"  # Replace with your JSON file path
    image_folder = "your_image_folder"  # Replace with your image folder path
    log_file = f"image_sizes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Setup logging
    logger = setup_logging(log_file)
    
    # Process images
    logger.info("Starting image processing...")
    sizes = process_images(json_path, image_folder, logger)
    
    if sizes:
        logger.info("Processing completed successfully!")
    else:
        logger.error("Processing failed!")

if __name__ == "__main__":
    main()