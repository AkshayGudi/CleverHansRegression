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
    """Given a data file, get all listed images, check if that image is present in our folder and  
    print their sizes"""
    
    try:
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get list of image files
        image_files = data['files']
        
        # Counters for summary
        total_images = len(image_files)
        processed_images = 0
        missing_images = 0
        
        logger.info(f"Starting to process {total_images} images...")
        
        # Process each image
        for img_name in image_files:
            img_path = os.path.join(image_folder, f"{img_name}.jpeg")
            
            if not os.path.exists(img_path):
                missing_images += 1
                error_msg = f"ERROR: Image not found: {img_path}"
                logger.error(error_msg)
                continue
            
            try:
                # Load image using PIL
                with Image.open(img_path) as img:
                    size = img.size  # (width, height)
                    format = img.format
                    mode = img.mode
                    
                    print("Processed_images - " + str(processed_images) + " out of  " + str(total_images))
                    print("\n")
                    logger.info(f"Processed: {img_name}")
                    logger.info(f"  Size (W×H): {size}")
                    logger.info(f"  Format: {format}")
                    logger.info(f"  Mode: {mode}")
                    logger.info("-" * 50)
                    
                    processed_images += 1
                    
            except Exception as img_error:
                error_msg = f"ERROR processing {img_path}: {str(img_error)}"
                logger.error(error_msg)
                
        # Log summary
        logger.info("\nProcessing Summary:")
        logger.info("=" * 50)
        logger.info(f"Total images in JSON: {total_images}")
        logger.info(f"Successfully processed: {processed_images}")
        logger.info(f"Missing images: {missing_images}")
        logger.info(f"Failed to process: {total_images - processed_images - missing_images}")

        return processed_images
                
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")

def main():
    # Setup paths
    json_path = "config/datasplit/temp_data_test.json"  # Replace with your JSON file path
    image_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/train"  # Replace with your image folder path
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