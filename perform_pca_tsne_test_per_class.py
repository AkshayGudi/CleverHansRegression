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
def mark_special_images(labels, filenames, special_set, special_label):
    new_labels = []
    for lbl, fname in zip(labels, filenames):
        if fname in special_set and lbl == special_label:  # only override label=1 subset
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


def main(params):
    # ==================== Step 1. Create dataset ====================
    dataset = MyDataModuleDiabRet(params)
    
    special_csv_path = '/sc/home/akshay.gudi/data_store/DR/without_yellow/test_yellow_patch.csv'
    # special_csv_path = '/sc/home/akshay.gudi/data_store/DR/yellow_train_data/test_yellow_patch.csv'
    tsne_sub_folder = '/test_without_art_tsne_pca_per_class1'
    # training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/DR_model_21_Aug/test-with-artifact/exp3/gpupro/Fold0_DR_exp4_21_Aug_4/'
    training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/DR_model_21_Aug/without-artifact/exp1/gpupro/Fold0_DR_exp4_21_Aug_5/'
    sub_folder = 'saved_models/Epoch_50_after_protopushing.pth'

    model_path = training_root_folder + sub_folder

    # ==================== Step 2. Extract features ====================
    feature_maps, prototypes, image_labels, prototype_labels, image_names = extract_features_and_prototypes(
        model_path,
        dataset.test_dataloader()
    )

    batch_size, channels, height, width = feature_maps.shape
    feature_maps_reshaped = feature_maps.reshape(batch_size, -1)

    print("Feature maps shape: ", feature_maps.shape)
    print("Prototype vectors shape: ", prototypes.shape)
    print("Number of images: ", len(image_labels))
    print("Number of prototypes: ", len(prototype_labels))

    # ==================== Step 3. Save extracted features ====================
    save_dir = Path(training_root_folder + tsne_sub_folder)
    save_dir.mkdir(exist_ok=True)

    np.save(save_dir / 'feature_maps.npy', feature_maps)
    np.save(save_dir / 'prototypes.npy', prototypes)
    np.save(save_dir / 'image_labels.npy', image_labels)
    np.save(save_dir / 'prototype_labels.npy', prototype_labels)
    np.save(save_dir / 'image_names.npy', image_names)

    print(f"Saved feature maps and prototypes to {save_dir}")

    # ==================== Step 4. Load special CSV ====================
    special_set = set(pd.read_csv(special_csv_path)["image_name"].astype(str).tolist())
    print(f"Loaded {len(special_set)} special images from {special_csv_path}")

    # Mark special images for Class 2 (but remember labels were shifted by -1, so class 2 → label=1)
    special_label = 1
    new_labels = mark_special_images(image_labels, image_names, special_set, special_label)

    # ==================== Step 5. Run PCA & t-SNE per class ====================
    per_class_save_dir = save_dir / "per_class"
    per_class_save_dir.mkdir(exist_ok=True)

    plot_pca_per_class(feature_maps_reshaped, new_labels, per_class_save_dir)
    plot_tsne_per_class(feature_maps_reshaped, new_labels, per_class_save_dir)

    print("Completed PCA and t-SNE per class plots")

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

    # special_csv = "/sc/home/akshay.gudi/data_store/DR/yellow_patch.csv"    

    # Parse command line arguments
    args_dict = vars(parser.parse_args())
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
        main(params)

        txt_logger.removeHandler(filehandler)