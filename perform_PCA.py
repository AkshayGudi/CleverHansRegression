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
    with torch.no_grad():
        for batch in tqdm(dataloader):
                images = batch[0].cuda()
                
                # Get feature maps using conv_features method
                features = ppnet_model.conv_features(images)   # Shape: [batch_size, 128, 9, 9]
                feature_maps.append(features.cpu().numpy())
                break

    # 4. Combine all feature maps
    all_feature_maps = np.concatenate(feature_maps, axis=0)

    # 5. Get prototype vectors
    prototype_vectors = ppnet_model.prototype_vectors.detach().cpu().numpy()

    return all_feature_maps, prototype_vectors

# Usage example:
def main(params):
     

    # Create dataset
    dataset = MyDataModuleDiabRet(params)

     # Your model path
    model_path = 'current_savedmodel_data/random_yellow/gpupro/Fold0_subset_yellow_22May_1/saved_models/Epoch_50_after_protopushing.pth'
     
    # Extract features and prototypes
    feature_maps, prototypes = extract_features_and_prototypes(
           model_path, 
           dataset.train_dataloader()
     )
     
    print("Feature maps shape:", feature_maps.shape)
     # Should be: [num_images, 128, 9, 9]
     
    print("Prototype vectors shape:", prototypes.shape)
     # Should be: [50, 128, 1, 1]
     
     # Optional: Save the extracted features
    save_dir = Path('extracted_features')
    save_dir.mkdir(exist_ok=True)
     
    np.save(save_dir / 'feature_maps.npy', feature_maps)
    np.save(save_dir / 'prototypes.npy', prototypes)

    print("Completed processing it")

# To visualize features (optional):
def visualize_features(feature_maps, prototypes):
     # 1. Visualize a feature map
    plt.figure(figsize=(10, 5))
     
    # Show first channel of first image
    plt.subplot(1, 2, 1)
    plt.imshow(feature_maps[0, 0])
    plt.colorbar()
    plt.title('First Channel of Feature Map')

    # Show prototype
    plt.subplot(1, 2, 2)
    plt.imshow(prototypes[0, :, 0, 0].reshape(8, 16))   # Reshape 128 to 8x16 for visualization
    plt.colorbar()
    plt.title('First Prototype Vector')

    plt.tight_layout()
    plt.show()

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

    