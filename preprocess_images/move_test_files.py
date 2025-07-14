import os
import shutil
import json

# Paths
src_folder = "/sc/home/akshay.gudi/coldstore/retinal_fundus/preprocessed_fundus"
test_dst_folder = "/sc/home/akshay.gudi/coldstore/retinal_fundus/preprocessed_fundus/test"
train_dst_folder = "/sc/home/akshay.gudi/coldstore/retinal_fundus/preprocessed_fundus/train"

# Ensure destination folders exist
os.makedirs(test_dst_folder, exist_ok=True)
os.makedirs(train_dst_folder, exist_ok=True)

# Load the JSON with test filenames
with open("/sc/home/akshay.gudi/code/CleverHansRegression/config/datasplit/brset_config/brset_test.json", "r") as f:
    test_data = json.load(f)

test_files = set(test_data["files"])

print(test_files)

# Move files
for file_name in os.listdir(src_folder):
    if file_name.lower().endswith('.jpg'):
        src_path = os.path.join(src_folder, file_name)
        
        if not os.path.isfile(src_path):
            continue  # skip non-files
        
        name_without_ext = os.path.splitext(file_name)[0]

        if name_without_ext in test_files:
            dst_path = os.path.join(test_dst_folder, file_name)
        else:
            dst_path = os.path.join(train_dst_folder, file_name)

        shutil.move(src_path, dst_path)
        print("Moved the file - " + str(file_name))

print("========================================= Completed =======================================")
