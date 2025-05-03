import json
import pandas as pd
import os
import logging
from datetime import datetime

def setup_logging(log_file='create_csvs.log'):
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

def create_fold_csvs(json_path, output_dir, logger):
    """Create CSV files for each fold's train and validation data"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Process each fold
        for fold_name, fold_data in data.items():
            logger.info(f"Processing {fold_name}...")
            
            # Process train and validation data
            for split in ['train', 'val']:
                # Get data for current split
                split_data = fold_data[split]
                
                # Create DataFrame
                df = pd.DataFrame({
                    'files': split_data['files'],
                    'labels': split_data['labels'],
                    'fussed_labels': split_data['fussed_labels']
                })
                
                # Create filename
                filename = f"{fold_name.lower().replace(' ', '_')}_{split}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Save to CSV
                df.to_csv(filepath, index=False)
                
                # Log statistics
                logger.info(f"Created {filename}")
                logger.info(f"  Rows: {len(df)}")
                logger.info(f"  Sample data:")
                logger.info(f"    First row: {df.iloc[0].to_dict()}")
                logger.info(f"    Last row: {df.iloc[-1].to_dict()}")
                logger.info("-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating CSVs: {str(e)}")
        return False

def verify_csvs(output_dir, logger):
    """Verify created CSV files"""
    try:
        logger.info("\nVerifying created CSV files:")
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        
        for csv_file in sorted(csv_files):
            filepath = os.path.join(output_dir, csv_file)
            df = pd.read_csv(filepath)
            
            logger.info(f"\nFile: {csv_file}")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Data types:\n{df.dtypes}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")

def main():
    # Setup paths
    json_path = "config/datasplit/new_data_cv.json"  # Replace with your JSON file path
    output_dir = "fold_csvs"
    log_file = f"create_csvs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Setup logging
    logger = setup_logging(log_file)
    
    # Create CSVs
    logger.info("Starting CSV creation...")
    success = create_fold_csvs(json_path, output_dir, logger)
    
    if success:
        logger.info("\nCSV creation completed successfully!")
        # Verify created files
        verify_csvs(output_dir, logger)
    else:
        logger.error("CSV creation failed!")

if __name__ == "__main__":
    main()