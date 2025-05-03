import pandas as pd
import glob
import os

# Define the input file pattern
input_files = 'fold_*_train.csv'

# Define the output file names
output_files = {
    1: 'label_1.csv',
    2: 'label_2.csv',
    3: 'label_3.csv',
    4: 'label_4.csv',
    5: 'label_5.csv'
}

# Initialize empty DataFrames for each label
label_dfs = {label: pd.DataFrame() for label in range(1, 6)}

# Read all input CSV files
for file in glob.glob(input_files):
    df = pd.read_csv(file)
    
    # Filter rows based on labels and append to corresponding DataFrame
    for label in range(1, 6):
        label_dfs[label] = label_dfs[label].append(df[df['labels'] == label], ignore_index=True)

# Write each label DataFrame to a separate CSV file
for label, output_file in output_files.items():
    if not label_dfs[label].empty:
        label_dfs[label].to_csv(output_file, index=False)
        print(f"Created {output_file}")
    else:
        print(f"No data for label {label}, skipping file creation")

print("Processing complete!")