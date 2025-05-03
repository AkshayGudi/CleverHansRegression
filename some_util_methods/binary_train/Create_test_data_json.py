import pandas as pd
import json
import numpy as np
import os

input_csv = "/dhc/home/akshay.gudi/code/CleverHansRegression/custom_data/test_data/class_2_test_yellow_patch.csv"

# Read the CSV file
df = pd.read_csv(input_csv)

df['files'] = df['files'].apply(lambda x: os.path.splitext(x)[0])

result = {}

result["files"] = df["files"].tolist()

result["labels"] = df["labels"].tolist()

# Convert numpy int64 to regular Python int for JSON serialization
def convert_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Save the result to a JSON file

output_file = "/dhc/home/akshay.gudi/code/CleverHansRegression/config/datasplit/class2_binary/kfold_class2_test.json"
with open(output_file, 'w') as f:
    json.dump(result, f, default=convert_to_python_types, indent=2)

print("K-Fold data has been saved to - ", output_file)