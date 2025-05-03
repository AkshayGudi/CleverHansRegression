import os

def find_extra_files(data_new_path, data_old_path):
    # Get list of files in both directories
    data_new_files = set(os.listdir(data_new_path))
    data_old_files = set(os.listdir(data_old_path))
    
    # Find files that are in data_new but not in data_old
    extra_files = data_new_files - data_old_files
    
    return extra_files

# Paths to the directories
data_new_path = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/preprocessed_Bgraham/test"
data_old_path = "/dhc/home/akshay.gudi/coldstore/diabetic_retino_data/data/test"

# Find and print the extra files
extra_files = find_extra_files(data_new_path, data_old_path)

print("Extra files in data_new:")
for file in sorted(extra_files):
    print(file)
