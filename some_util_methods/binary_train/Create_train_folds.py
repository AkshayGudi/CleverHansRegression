import pandas as pd
import json
from sklearn.model_selection import KFold
import numpy as np
import os

# Read the CSV file
df = pd.read_csv('/dhc/home/akshay.gudi/code/CleverHansRegression/custom_data/train_data/class_2_train_yellow_patch.csv')

df['files'] = df['files'].apply(lambda x: os.path.splitext(x)[0])

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the result dictionary
result = {}

# Perform K-Fold split
for fold, (train_index, val_index) in enumerate(kf.split(df)):
    train_data = df.iloc[train_index]
    val_data = df.iloc[val_index]
    
    result[f"Fold {fold}"] = {
        "train": {
            "files": train_data['files'].tolist(),
            "labels": train_data['labels'].tolist(),
            "fussed_labels": train_data['fussed_labels'].tolist()
        },
        "val": {
            "files": val_data['files'].tolist(),
            "labels": val_data['labels'].tolist(),
            "fussed_labels": val_data['fussed_labels'].tolist()
        }
    }

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

output_file = "/dhc/home/akshay.gudi/code/CleverHansRegression/config/datasplit/class2_binary/kfold_class2_train.json"
with open(output_file, 'w') as f:
    json.dump(result, f, default=convert_to_python_types, indent=2)

print("K-Fold data has been saved to - ", output_file)