import pandas as pd

def extract_image_names(input_csv, output_csv):
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    # Select only the image_name column
    df_image = df[['image_name']].copy()
    
    # Add ".jpeg" extension to each image_name
    df_image['image_name'] = df_image['image_name'].astype(str) + '.jpeg'
    
    # Save to new CSV (no index column)
    df_image.to_csv(output_csv, index=False)

# Example usage
root_folder = "/sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14/data_details_class3/"
input_csv = root_folder + "DR_train_data_patch.csv"
output_csv = root_folder + "train_yellow_patch.csv"
extract_image_names(input_csv, output_csv)

input_csv = root_folder + "DR_test_data_patch.csv"
output_csv = root_folder + "test_yellow_patch.csv"
extract_image_names(input_csv, output_csv)
