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
from mpl_toolkits.mplot3d import Axes3D

def extract_features_and_prototypes(model_path, dataloader):
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
    ppnet_model = ppnet_model.cuda()   # If using GPU

    # 3. Extract features
    feature_maps = []
    image_labels = [] # store labels

    with torch.no_grad():
        for batch in tqdm(dataloader):
                images = batch[0].cuda()
                labels = batch[1] # Get labels from batch
                
                # Get feature maps using conv_features method
                features = ppnet_model.conv_features(images)   # Shape: [batch_size, 128, 9, 9]
                feature_maps.append(features.cpu().numpy())
                image_labels.extend(labels.cpu().numpy()) # Store labels
                # break

    # 4. Combine all feature maps
    all_feature_maps = np.concatenate(feature_maps, axis=0)   

    # 5. Get prototype vectors
    prototype_vectors = ppnet_model.prototype_vectors.detach().cpu().numpy()

    prototype_labels = ppnet_model.proto_classes.cpu().numpy()

    return all_feature_maps, prototype_vectors, np.array(image_labels), prototype_labels

def perform_and_plot_pca(data, labels, title, save_path):
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data)

    # Plotting
    fig = plt.figure(figsize=(20, 15))

    # Scatter plot of first two PCs
    ax1 = fig.add_subplot(2, 2, 1)
    scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap="viridis")

    # ax1.scatter(pca_result[:, 0], pca_result[:, 1])
    ax1.set_title(f'{title}: First Two Principal Components')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax1, label='Severity Level')

    # Explained variance ratio
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(np.cumsum(pca.explained_variance_ratio_))
    ax2.set_title('Cumulative Explained Variance Ratio')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')

    # Loadings of first PC
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pca.components_[0])
    ax3.set_title('Loadings of First Principal Component')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Loading Value')

    # 3D scatter plot of first three PCs
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    scatter3d = ax4.scatter(pca_result[:, 0],
                            pca_result[:, 1],
                            pca_result[:, 2],
                            c=labels, cmap='viridis')


    # ax4.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
    ax4.set_title(f'{title}: First Three Principal Components')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_zlabel('PC3')
    plt.colorbar(scatter3d, ax=ax4, label='Severity Level')

    plt.tight_layout()
    
    # Save the figure
    filename = f'pca_analysis_{title.lower().replace(" ", "_")}.png'
    save_file_path = save_path / filename
    plt.savefig(save_file_path)
    # plt.savefig(save_path / f'pca_analysis_{title.lower().replace(" ", "_")}.png')
    plt.close()

    # print(f"PCA analysis plot for {title} saved to {save_path / f'pca_analysis_{title.lower().replace(' ', '_')}.png'}")
    print(f"PCA analysis plot for {title} saved to {save_file_path}")


def perform_separate_pcas(feature_maps, prototypes, image_labels, prototype_labels, save_path):
    # Reshape feature maps: each image is a data point
    batch_size, channels, height, width = feature_maps.shape
    feature_maps_reshaped = feature_maps.reshape(batch_size, -1)

    # Reshape prototypes: each prototype is a data point
    prototypes_reshaped = prototypes.reshape(prototypes.shape[0], -1)

    # Perform PCA on feature maps
    perform_and_plot_pca(feature_maps_reshaped, image_labels, "Feature Maps", save_path)

    # Perform PCA on prototypes
    perform_and_plot_pca(prototypes_reshaped, prototype_labels, "Prototypes", save_path)


def main(params):

    # Create dataset
    dataset = MyDataModuleDiabRet(params)

     # Your model path
    model_path = 'current_savedmodel_data/random_yellow/gpupro/Fold0_subset_yellow_22May_1/saved_models/Epoch_50_after_protopushing.pth'
     
    # Extract features and prototypes
    feature_maps, prototypes, image_labels, prototype_labels = extract_features_and_prototypes(
        model_path,
        dataset.train_dataloader()
    )
    
    print("Feature maps shape: ", feature_maps.shape)
    print("Prototype vectors shape: ", prototypes.shape)
    print("Number of images: ", len(image_labels))
    print("Number of prototypes: ", len(prototype_labels))
    
    # Optional: Save the extracted features
    save_dir = Path('extracted_features')
    save_dir.mkdir(exist_ok=True)
    
    np.save(save_dir / 'feature_maps.npy', feature_maps)
    np.save(save_dir / 'prototypes.npy', prototypes)
    np.save(save_dir / 'image_labels.npy', image_labels)
    np.save(save_dir / 'prototype_labels.npy', prototype_labels)

    # Perform separate PCAs on feature maps and prototypes
    perform_separate_pcas(feature_maps, prototypes, image_labels, prototype_labels, save_dir)

    print("Completed processing it")


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

    # Parse command line arguments
    args_dict = vars(parser.parse_args())
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = { **params_dict,**args_dict}
    params = Parameters.from_dict(params_exp)

    params.set_savepaths()
    base_runname = params.run_name

    num_folds = 1
    for fold in range(num_folds):
        # Set crossvalidation
        params.cv_fold = fold
        
        # Reinitialize savepaths
        params.run_name = f'Fold{fold}_{base_runname}'
        params.set_savepaths()
        params.save_path_ims.mkdir(parents=True, exist_ok=True)

        # Set handler to new file
            # logging for outside trainer object
        txt_logger = logging.getLogger("pytorch_lightning")
        filehandler = logging.FileHandler(params.save_path / "logfile.log")
        txt_logger.addHandler(filehandler)

        # Run training
        rank_zero_info(f'Start fold {fold}')
        
        main(params)

        # Remove handler from logger
        txt_logger.removeHandler(filehandler)
