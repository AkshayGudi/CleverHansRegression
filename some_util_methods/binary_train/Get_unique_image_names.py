import pandas as pd

def filter_unique_files(csv_path, output_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Drop duplicates, keeping only the first occurrence
    df_unique = df.drop_duplicates(subset=['files'], keep='first')
    
    # Save the filtered data to a new CSV file
    df_unique.to_csv(output_path, index=False)
    
    print(f"Filtered data saved to {output_path}")

# Example usage
input_csv = "/dhc/home/akshay.gudi/code/CleverHansRegression/fold_csvs/extracted_by_label/label_2.csv"  # Replace with your input CSV file
output_csv = "/dhc/home/akshay.gudi/code/CleverHansRegression/fold_csvs/extracted_by_label/unique_class_2.csv"  # Replace with your desired output file
filter_unique_files(input_csv, output_csv)
