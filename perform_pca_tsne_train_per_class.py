import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from insight_training import model
from define_parameters import NetworkParams
from define_parameters import Parameters
from helpers import load_json
from datamodule import  MyDataModuleDiabRet
import matplotlib.pyplot as plt
import argparse
import logging
from pytorch_lightning.utilities import rank_zero_info
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys

def extract_features_and_prototypes(model_path, dataloader):
    print("Extracting features.........")
    # 1. Initialize model
    network_params = NetworkParams()   # Your network parameters
    ppnet_model = model.construct_PPNet(network_params=network_params)

    # 2. Load your trained model
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        ppnet_model.load_state_dict(checkpoint['state_dict'])
    else:
        ppnet_model.load_state_dict(checkpoint)

    ppnet_model.eval()
    ppnet_model = ppnet_model.cuda()  # If using GPU

    # 3. Extract features
    feature_maps, image_labels, image_names = [], [], []

    with torch.no_grad():
        i = 0
        for i, batch in enumerate(tqdm(dataloader)):
            # Expecting dataloader to return (images, labels, filenames)
            images, labels, filenames = batch
            images = images.cuda()            

            # Get feature maps using conv_features method
            features = ppnet_model.conv_features(images)   # Shape: [batch_size, 128, 9, 9]
            feature_maps.append(features.cpu().numpy())
            image_labels.extend(labels.cpu().numpy()) # Store labels
            image_names.extend(filenames)
            
            print(f"extracting for batch - {i+1}")                

    # 4. Combine
    all_feature_maps = np.concatenate(feature_maps, axis=0)
    prototype_vectors = ppnet_model.prototype_vectors.detach().cpu().numpy()
    prototype_labels = ppnet_model.proto_classes.cpu().numpy()

    image_labels = [x - 1 for x in image_labels]
    # 1. Clip values at 5.0
    clipped = np.clip(prototype_labels, None, 5.0)

    # 2. Convert to integers with ceiling (round up)
    ceil_vals = np.ceil(clipped).astype(int)

    # 3. Subtract 1 (final values range from 0 to 4)
    prototype_labels = ceil_vals - 1

    return (
        all_feature_maps,
        prototype_vectors,
        np.array(image_labels),
        prototype_labels,
        np.array(image_names),
    )

# ====================== Mark Special Images ======================
def mark_special_images(labels, filenames, special_set, class_with_artifact):
    new_labels = []
    for lbl, fname in zip(labels, filenames):
        if fname in special_set and lbl == class_with_artifact:  # only override label=1 subset
            new_labels.append(5)  # special new class
        else:
            new_labels.append(lbl)
    return np.array(new_labels)

def plot_pca_per_class(data, labels, save_path, special_label=1):
    """
    Plot PCA separately for each class.
    For special_label (e.g., class 2), split into normal vs special subset.
    """
    pca = PCA()
    pca_result = pca.fit_transform(data)

    unique_classes = np.unique(labels)

    for class_id in unique_classes:
        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, 1)
        if class_id == special_label or class_id == 5:  # handle patched class separately
            mask_normal = labels == special_label
            mask_special = labels == 5  # patched subset
            ax1.scatter(pca_result[mask_normal, 0], pca_result[mask_normal, 1],
                        c='green', label=f'Class {special_label+1} (normal)')
            ax1.scatter(pca_result[mask_special, 0], pca_result[mask_special, 1],
                        c='yellow', label=f'Class {special_label+1} (yellow patch)')
        else:
            mask = labels == class_id
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1],
                        label=f'Class {class_id+1}')

        ax1.set_title("PCA 2D")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.legend()

        # Explained variance
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(np.cumsum(pca.explained_variance_ratio_))
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_xlabel("Components")
        ax2.set_ylabel("Variance Ratio")

        # PC1 loadings
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(pca.components_[0])
        ax3.set_title("PC1 Loadings")

        # 3D Scatter
        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        if class_id == special_label or class_id == 5:
            ax4.scatter(pca_result[mask_normal, 0], pca_result[mask_normal, 1],
                        pca_result[mask_normal, 2], c='green', label="Normal")
            ax4.scatter(pca_result[mask_special, 0], pca_result[mask_special, 1],
                        pca_result[mask_special, 2], c='yellow', label="Yellow patch")
        else:
            ax4.scatter(pca_result[mask, 0], pca_result[mask, 1],
                        pca_result[mask, 2], label=f'Class {class_id+1}')

        ax4.set_title("PCA 3D")
        ax4.set_xlabel("PC1")
        ax4.set_ylabel("PC2")
        ax4.set_zlabel("PC3")
        ax4.legend()

        plt.tight_layout()
        save_file = save_path / f"class_{class_id+1}_pca.png"
        plt.savefig(save_file)
        plt.close()
        print(f"Saved PCA for class {class_id+1} to {save_file}")


def plot_tsne_per_class(data, labels, save_path, special_label=1):
    """
    Plot t-SNE separately for each class.
    For special_label (e.g., class 2), split into normal vs special subset.
    """
    print("Started TSNE..")
    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(data)
    unique_classes = np.unique(labels)

    print("Plotting TSNE..")
    for class_id in unique_classes:
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(1, 2, 1)
        if class_id == special_label or class_id == 5:
            mask_normal = labels == special_label
            mask_special = labels == 5
            ax1.scatter(tsne_result[mask_normal, 0], tsne_result[mask_normal, 1],
                        c='green', label=f'Class {special_label+1} (normal)')
            ax1.scatter(tsne_result[mask_special, 0], tsne_result[mask_special, 1],
                        c='yellow', label=f'Class {special_label+1} (yellow patch)')
        else:
            mask = labels == class_id
            ax1.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                        label=f'Class {class_id+1}')

        ax1.set_title("t-SNE 2D")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        if class_id == special_label or class_id == 5:
            ax2.scatter(tsne_result[mask_normal, 0], tsne_result[mask_normal, 1],
                        tsne_result[mask_normal, 2], c='green', label="Normal")
            ax2.scatter(tsne_result[mask_special, 0], tsne_result[mask_special, 1],
                        tsne_result[mask_special, 2], c='yellow', label="Yellow patch")
        else:
            ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                        tsne_result[mask, 2], label=f'Class {class_id+1}')

        ax2.set_title("t-SNE 3D")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.set_zlabel("t-SNE 3")
        ax2.legend()

        plt.tight_layout()
        save_file = save_path / f"class_{class_id+1}_tsne.png"
        plt.savefig(save_file)
        plt.close()
        print(f"Saved t-SNE for class {class_id+1} to {save_file}")


def main(params, use_saved=True):
    
    # training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/DR_model_21_Aug/with-artifact/exp1/gpupro/Fold0_DR_exp4_21_Aug_2/'

    # Path of csv file which has names of all images with yellow patch
    images_with_artifact_csv_path = "/sc/home/akshay.gudi/data_store/DR/yellow_train_data/data_details_50_train/train_yellow_patch.csv"
    
    # path where all training results including trained model is stored
    training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/DR_model_3_Sept/with-artifact/exp1/gpupro/Fold0_DR_exp1_03_Sept_1/'

    #Path to save the TSNE PCA results
    pca_tsne_sub_folder = '/train_tsne_pca_with_artifact_e20'
    
    # sub folder under training_root_folder where trained model is stored
    # model_path_sub_folder = 'saved_models/Epoch_50_after_protopushing.pth'
    model_path_sub_folder = 'saved_models/Epoch_20_after_protopushing.pth'

    model_path = training_root_folder + model_path_sub_folder

    # Directory where features are stored or will be stored
    save_dir = Path(training_root_folder + pca_tsne_sub_folder)
    save_dir.mkdir(exist_ok=True)    

    # --- Decide whether to load or extract ---
    if use_saved:
        feature_file = save_dir / "feature_maps.npy"
        proto_file   = save_dir / "prototypes.npy"
        img_lbl_file = save_dir / "image_labels.npy"
        proto_lbl_file = save_dir / "prototype_labels.npy"
        img_name_file  = save_dir / "image_names.npy"

        if all(f.exists() for f in [feature_file, proto_file, img_lbl_file, proto_lbl_file, img_name_file]):
            print("✅ Using previously saved feature maps and prototypes.")
            feature_maps = np.load(feature_file)
            prototypes = np.load(proto_file)
            image_labels = np.load(img_lbl_file)
            prototype_labels = np.load(proto_lbl_file)
            image_names = np.load(img_name_file)
        else:
            print("❌ Error: One or more saved .npy files not found. Please rerun with --use_saved False")
            sys.exit(1)
    else:
        print("🔄 Extracting features and prototypes from model...")
        
        # Create dataset
        dataset = MyDataModuleDiabRet(params)

        # Extract features and prototypes
        feature_maps, prototypes, image_labels, prototype_labels, image_names = extract_features_and_prototypes(
            model_path,
            dataset.train_dataloader()        
        )

        batch_size, channels, height, width = feature_maps.shape
        feature_maps_reshaped = feature_maps.reshape(batch_size, -1)

        print("Feature maps shape: ", feature_maps.shape)
        print("Prototype vectors shape: ", prototypes.shape)
        print("Number of images: ", len(image_labels))
        print("Number of prototypes: ", len(prototype_labels))

        np.save(save_dir / 'feature_maps.npy', feature_maps)
        np.save(save_dir / 'prototypes.npy', prototypes)
        np.save(save_dir / 'image_labels.npy', image_labels)
        np.save(save_dir / 'prototype_labels.npy', prototype_labels)
        np.save(save_dir / 'image_names.npy', image_names)

        print(f"Saved feature maps and prototypes to {save_dir}")

    # ==================== Step 4. Load special CSV ====================
    images_with_artifact = set(pd.read_csv(images_with_artifact_csv_path)["image_name"].astype(str).tolist())
    print(f"Loaded {len(images_with_artifact)} special images from {images_with_artifact_csv_path}")

    # Mark special images for Class 2 (but remember labels were shifted by -1, so class 2 → label=1)
    new_labels = mark_special_images(image_labels, image_names, images_with_artifact, class_with_artifact=1)

    # ==================== Step 5. Run PCA & t-SNE per class ====================
    per_class_save_dir = save_dir / "per_class"
    per_class_save_dir.mkdir(exist_ok=True)

    plot_pca_per_class(feature_maps_reshaped, new_labels, per_class_save_dir)
    plot_tsne_per_class(feature_maps_reshaped, new_labels, per_class_save_dir)

    print("Completed PCA and t-SNE per class plots")

def get_flag_value(flag, default=True):
    """Extract value of a CLI flag like --use_saved=True/False"""
    for arg in sys.argv:
        if arg.startswith(flag + "="):
            val = arg.split("=", 1)[1].lower()
            if val in ["true", "1", "yes"]:
                return True
            elif val in ["false", "0", "no"]:
                return False
    return default


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Run ProtoPnet code')

    parser.add_argument('--run_name',
                help = 'Define the run_name for saving everything (example: test_resnet)',
                default='DummyName')
                
    parser.add_argument('--param_jsonpath',
                        default='config/params_example_ordinal.json',
                        required=True,
                        help='Define the parameter file used for training (example: config/params_example_ordinal.json)')
    
    parser.add_argument('--datapath',
                        required=True)
    
    parser.add_argument('--savepath',
                        default = 'savedmodel',
                        required=True)
                    
    parser.add_argument('--pretrained_path',
                        default = 'config/pretrained_model.ckpt')

    # NEW FLAG
    # ✅ Extract custom flag manually
    use_saved_flag = get_flag_value("--use_saved", default=True)
    print(f"use_saved = {use_saved_flag}")

    # Parse command line arguments
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = { **params_dict,**args_dict}
    params = Parameters.from_dict(params_exp)

    params.set_savepaths()
    base_runname = params.run_name

    num_folds = 1
    for fold in range(num_folds):
        params.cv_fold = fold
        params.run_name = f'Fold{fold}_{base_runname}'
        params.set_savepaths()
        params.save_path_ims.mkdir(parents=True, exist_ok=True)

        txt_logger = logging.getLogger("pytorch_lightning")
        filehandler = logging.FileHandler(params.save_path / "logfile.log")
        txt_logger.addHandler(filehandler)

        rank_zero_info(f'Start fold {fold}')        
        main(params, use_saved=use_saved_flag)

        txt_logger.removeHandler(filehandler)