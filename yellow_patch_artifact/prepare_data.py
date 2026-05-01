import pandas as pd
import os

# Extract 80% of image names for given label where yellow patch can be applied
def prepare_patch_data(input_csv, output_base_path, artifact_labeled_file, yellow_patch_file, class_label):

    os.makedirs(output_base_path, exist_ok=True)

    # Read the CSV
    df = pd.read_csv(input_csv)

    # Filter label = 5
    label2_df = df[df["labels"] == class_label]

    # Take 80% sample (random)
    sample_df = label2_df.sample(frac=1.0, random_state=42)  # random_state for reproducibility

    selected_images = sample_df['image_name'].tolist()
    # Create artifact_label column: 1 if in selected_images, else 0
    df['artifact_label'] = df['image_name'].apply(lambda x: 1 if x in selected_images else 0)

    # Create output CSV with image_name and artifact_label
    df[['image_name', 'artifact_label']].to_csv(output_base_path + artifact_labeled_file, index=False)

    # Save to new CSV
    sample_df.to_csv(output_base_path + yellow_patch_file, index=False)

    print(f"Saved {len(sample_df)} rows to {output_base_path + yellow_patch_file}")

    return output_base_path + artifact_labeled_file


if __name__ == "__main__":

    # For train images
    train_file_path = "/sc/home/akshay.gudi/data_store/DR/DR_train_data.csv"
    base_path = "/sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14/data_details_class3/"
    
    # this file will be created newly, which has images labeled with and without yellow patch
    train_artifact_labeled_file = "train_labeled_data.csv"

    # this file will be created newly, which has only images name with yellow patch
    train_yellow_file = "DR_train_data_patch.csv"
    class_label = 3

    train_labeled_file_path = prepare_patch_data(train_file_path, base_path, train_artifact_labeled_file, train_yellow_file, class_label)
    print("Train labeled file path - " + train_labeled_file_path)

    # For test images
    test_file_path = "/sc/home/akshay.gudi/data_store/DR/DR_test_data.csv"
    # output_test_base_path = "/sc/home/akshay.gudi/data_store/DR/each_class_yellow/class1/"

    # this file will be created newly, which has images labeled with and without yellow patch
    test_artifact_labeled_file = "test_labeled_data.csv"

    # this file will be created newly, which has only images name with yellow patch
    test_yellow_file = "DR_test_data_patch.csv"
    test_labeled_file_path = prepare_patch_data(test_file_path, base_path, test_artifact_labeled_file, test_yellow_file, class_label)
    print("Test labeled file path - " + test_labeled_file_path)