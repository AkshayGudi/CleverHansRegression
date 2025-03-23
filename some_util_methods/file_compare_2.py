import os
from pathlib import Path
import logging
from datetime import datetime
from collections import deque

def setup_logging(log_file='extra_files.log'):
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

def compare_files_two_pointers(new_folder, old_folder, logger):
    """Compare files using two pointers approach"""
    try:
        # Get sorted lists of files
        new_files = sorted([f for f in os.listdir(new_folder) if f.endswith('.jpeg')])
        old_files = sorted([f for f in os.listdir(old_folder) if f.endswith('.jpeg')])
        
        # Convert to deque for efficient popping
        new_deque = deque(new_files)
        old_deque = deque(old_files)
        
        # List to store extra files
        extra_files = []
        old_extra_files = []
        
        # Log initial counts
        logger.info(f"Initial files in new folder: {len(new_files)}")
        logger.info(f"Initial files in old folder: {len(old_files)}")
        
        # Compare files using two pointers
        while new_deque and old_deque:
            new_file = new_deque[0]
            old_file = old_deque[0]
            
            logger.debug(f"Comparing: {new_file} with {old_file}")
            
            if new_file == old_file:
                # Files match, pop both
                new_deque.popleft()
                old_deque.popleft()
                logger.debug(f"Match found: {new_file}")
            elif new_file < old_file:
                # New file is extra
                extra_files.append(new_deque.popleft())
                logger.debug(f"Extra file found: {new_file}")
            else:
                # Old file not in new, skip it
                old_extra_files.append(old_deque.popleft())
                logger.debug(f"Skipping old file: {old_file}")
        
        # Add remaining files from new_deque
        extra_files.extend(new_deque)
        old_extra_files.extend(old_deque)
        
        # Log results
        logger.info(f"\nNumber of extra files found: {len(extra_files)}")
        logger.info(f"\nNumber of extra files in old folder found: {len(old_extra_files)}")
        
        if extra_files:
            logger.info("\nExtra files in new folder:")
            for idx, file in enumerate(extra_files, 1):
                file_path = os.path.join(new_folder, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                logger.info(f"{idx}. {file} ({size_mb:.2f} MB)")
        else:
            logger.info("\nNo extra files found in new folder.")

        if old_extra_files:
            logger.info("\nExtra files in old folder:")
            for idx, file in enumerate(old_extra_files, 1):
                file_path = os.path.join(old_folder, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                logger.info(f"{idx}. {file} ({size_mb:.2f} MB)")
        else:
            logger.info("\nNo extra files found in old folder.")            
            
        return extra_files
        
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        return None

def main():
    # Setup paths
    new_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/train"  # Replace with your new folder path
    old_folder = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/data/train"  # Replace with your old folder path
    log_file = f"extra_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Setup logging
    logger = setup_logging(log_file)
    
    # Process folders
    logger.info("Starting comparison using two pointers approach...")
    logger.info(f"New folder: {new_folder}")
    logger.info(f"Old folder: {old_folder}")
    
    extra_files = compare_files_two_pointers(new_folder, old_folder, logger)
    
    if extra_files is not None:
        logger.info("\nComparison completed successfully!")
        
        # Optional: Print summary
        logger.info("\nSummary:")
        logger.info(f"Total extra files: {len(extra_files)}")
        if extra_files:
            logger.info(f"First extra file: {extra_files[0]}")
            logger.info(f"Last extra file: {extra_files[-1]}")

if __name__ == "__main__":
    main()