import json
import csv
import os

# Read the JSON file
with open('/dhc/home/akshay.gudi/code/CleverHansRegression/config/datasplit/new_data_test.json', 'r') as json_file:
	data = json.load(json_file)

# Extract files and labels
files = data['files']
labels = data['labels']

# Create a dictionary to store files for each label
label_files = {}

# Group files by their labels
for file, label in zip(files, labels):
	if label not in label_files:
		label_files[label] = []
	label_files[label].append(file)

root_path = "/dhc/home/akshay.gudi/code/CleverHansRegression/custom_data/test_data"

# Create a CSV file for each label
for label, files in label_files.items():
	csv_filename = root_path + f'Test_class_{label}.csv'

	with open(csv_filename, 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)

		# Write the header
		csvwriter.writerow(['files', 'labels'])

		# Write the data
		for file in files:
			csvwriter.writerow([file, label])

	print(f"Created {csv_filename} with {len(files)} entries.")

print("Processing complete!")