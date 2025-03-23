import pandas as pd
import glob

# Define the input file pattern
input_files = 'label_*.csv'

# Process each input file
for file in glob.glob(input_files):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract unique values from the 'files' column
    unique_files = df['files'].unique()
    
    # Generate the output file name
    output_file = f"file_{file.replace('label_', '').replace('.csv', '.txt')}"
    
    # Write unique values to the output text file
    with open(output_file, 'w') as f:
        for unique_file in unique_files:
            f.write(f"{unique_file}\n")
    
    print(f"Created {output_file} with {len(unique_files)} unique entries")

print("Processing complete!")