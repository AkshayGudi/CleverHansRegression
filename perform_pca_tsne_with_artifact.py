import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from insight_training import model
from define_parameters import NetworkParams, Parameters
from helpers import load_json
from datamodule import MyDataModuleDiabRet
import matplotlib.pyplot as plt
import argparse
import logging
from pytorch_lightning.utilities import rank_zero_info
from sklearn.decomposition import PCA
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D


# ====================== Feature Extraction ======================
def extract_features_and_prototypes(model_path, dataloader):
    print("Extracting features.........")
    # 1. Initialize model
    network_params = NetworkParams()
    ppnet_model = model.construct_PPNet(network_params=network_params)

    # 2. Load trained model
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        ppnet_model.load_state_dict(checkpoint['state_dict'])
    else:
        ppnet_model.load_state_dict(checkpoint)

    ppnet_model.eval().cuda()

    # 3. Extract features
    feature_maps, image_labels, image_names = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # Expecting dataloader to return (images, labels, filenames)
            images, labels, filenames = batch
            images = images.cuda()

            features = ppnet_model.conv_features(images)
            feature_maps.append(features.cpu().numpy())
            image_labels.extend(labels.cpu().numpy())
            image_names.extend(filenames)   # keep filenames

            print(f"extracting for batch - {i+1}")

    # 4. Combine
    all_feature_maps = np.concatenate(feature_maps, axis=0)
    prototype_vectors = ppnet_model.prototype_vectors.detach().cpu().numpy()
    prototype_labels = ppnet_model.proto_classes.cpu().numpy()

    return (
        all_feature_maps,
        prototype_vectors,
        np.array(image_labels),
        prototype_labels,
        np.array(image_names),
    )


# ====================== Mark Special Images ======================
def mark_special_images(labels, filenames, special_set):
    new_labels = []
    for lbl, fname in zip(labels, filenames):
        print("file name is - ")
        print(fname)
        if fname in special_set and lbl == 1:  # only override label=1 subset
            new_labels.append(5)  # special new class
        else:
            new_labels.append(lbl)
    return np.array(new_labels)


# ====================== PCA + Plotting ======================
def perform_and_plot_pca(data, labels, title, save_path, class_colors=None):
    pca = PCA()
    pca_result = pca.fit_transform(data)

    # Colors for each class
    if class_colors is None:
        class_colors = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'yellow',
            4: 'orange',
            5: 'purple'   # special subset
        }

    class_names = {
        0: 'Class 1',
        1: 'Class 2',
        2: 'Class 3',
        3: 'Class 4',
        4: 'Class 5',
        5: 'Special patched Class 2'
    }

    fig = plt.figure(figsize=(20, 15))

    # 2D scatter
    ax1 = fig.add_subplot(2, 2, 1)
    for class_id in np.unique(labels):
        mask = labels == class_id
        ax1.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=class_colors[int(class_id)],
            label=class_names[int(class_id)],
        )
    ax1.set_title(f'{title}: First Two Principal Components')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()

    # Explained variance
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(np.cumsum(pca.explained_variance_ratio_))
    ax2.set_title('Cumulative Explained Variance Ratio')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')

    # Loadings of PC1
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pca.components_[0])
    ax3.set_title('Loadings of First Principal Component')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Loading Value')

    # 3D scatter
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for class_id in np.unique(labels):
        mask = labels == class_id
        ax4.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            pca_result[mask, 2],
            c=class_colors[int(class_id)],
            label=class_names[int(class_id)],
        )
    ax4.set_title(f'{title}: First Three Principal Components')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_zlabel('PC3')
    ax4.legend()

    plt.tight_layout()
    save_file_path = save_path / f'pca_analysis_{title.lower().replace(" ", "_")}.png'
    plt.savefig(save_file_path)
    plt.close()
    print(f"PCA analysis plot for {title} saved to {save_file_path}")


def perform_pca_and_tsne(feature_maps, image_labels, prototypes, prototype_labels, save_path):
    # Flatten feature maps
    batch_size, channels, height, width = feature_maps.shape
    feature_maps_reshaped = feature_maps.reshape(batch_size, -1)

    # Flatten prototypes
    prototypes_reshaped = prototypes.reshape(prototypes.shape[0], -1)

    pca_save_dir = save_path / "pca"
    pca_save_dir.mkdir(exist_ok=True)

    perform_and_plot_pca(
        feature_maps_reshaped,
        image_labels,
        "Feature Maps - PCA",
        pca_save_dir,
    )


# ====================== Main ======================
def main(params, special_csv_path):
    # Dataset
    dataset = MyDataModuleDiabRet(params)

    # Model path
    training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/training_results/DR_model_23_Jul/with-artifcat/exp1/gpupro/Fold0_DR_exp4_23_jul_1/'
    sub_folder = 'saved_models/Epoch_50_after_protopushing.pth'
    model_path = training_root_folder + sub_folder

    # Extract features
    feature_maps, prototypes, image_labels, prototype_labels, image_names = extract_features_and_prototypes(
        model_path,
        dataset.train_dataloader()
    )

    print("Feature maps shape: ", feature_maps.shape)
    print("Prototype vectors shape: ", prototypes.shape)
    print("Number of images: ", len(image_labels))
    print("Number of prototypes: ", len(prototype_labels))

    # Save extracted features
    save_dir = Path(training_root_folder + '/tsne_pca3')
    save_dir.mkdir(exist_ok=True)
    np.save(save_dir / 'feature_maps.npy', feature_maps)
    np.save(save_dir / 'prototypes.npy', prototypes)
    np.save(save_dir / 'image_labels.npy', image_labels)
    np.save(save_dir / 'prototype_labels.npy', prototype_labels)
    np.save(save_dir / 'image_names.npy', image_names)

    # Load special CSV
    special_set = set(pd.read_csv(special_csv_path)["image_name"].astype(str).tolist())
    print(f"Loaded {len(special_set)} special images from {special_csv_path}")

    # Mark special ones
    new_labels = mark_special_images(image_labels, image_names, special_set)

    # PCA
    perform_pca_and_tsne(feature_maps, new_labels, prototypes, prototype_labels, save_dir)

    print("Completed processing it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ProtoPnet PCA analysis with special CSV marking')

    parser.add_argument('--run_name',
                        help='Define run_name for saving everything',
                        default='DummyName')

    parser.add_argument('--param_jsonpath',
                        default='config/params_example_ordinal.json',
                        required=True,
                        help='Parameter file used for training')

    parser.add_argument('--datapath',
                        required=True)

    parser.add_argument('--savepath',
                        default='savedmodel',
                        required=True)

    parser.add_argument('--pretrained_path',
                        default='config/pretrained_model.ckpt')

    # parser.add_argument('--special_csv',
    #                     required=True,
    #                     help='CSV file with special yellow-patch images')

    special_csv = "/sc/home/akshay.gudi/data_store/DR/yellow_patch.csv"

    args_dict = vars(parser.parse_args())
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = {**params_dict, **args_dict}
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
        main(params, special_csv)

        txt_logger.removeHandler(filehandler)
