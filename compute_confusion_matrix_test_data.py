"""
Compute confusion matrix and accuracy on the TEST set.
Takes folder root (training run folder) as input, loads model and test loader,
outputs confusion matrix, accuracy, and a CSV with image_name, true_label, predicted_label.
Labels in CSV are 1-based (1-5) to match the dataset.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from define_parameters import NetworkParams, Parameters
from helpers import load_json, plot_confmatrix
from datamodule import MyDataModuleDiabRet
from insight_training import model
import pandas as pd


def load_model_and_predict(model_path, dataloader, params, device="cuda"):
    """Load PPNet from checkpoint and run inference on dataloader.
    Returns lists: image_names, true_labels_1based, pred_labels_1based, true_0based, pred_0based.
    """
    network_params = NetworkParams()
    ppnet = model.construct_PPNet(network_params=network_params)
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        ppnet.load_state_dict(checkpoint["state_dict"])
    else:
        ppnet.load_state_dict(checkpoint)
    ppnet.eval()
    ppnet = ppnet.to(device)

    min_label = getattr(params, "min_label", 1)
    max_label = getattr(params, "max_label", 5)

    image_names = []
    true_1 = []
    pred_1 = []
    pred_raw = []  # prediction before rounding/clamping

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting (test)"):
            images, labels, filenames = batch
            images = images.to(device)
            output, _, _ = ppnet(images, return_convs=False)
            pred_reg = output.squeeze().cpu()
            pred_raw.extend(pred_reg.numpy().tolist())
            pred_cls = torch.clamp(
                torch.round(pred_reg).long(),
                min=min_label,
                max=max_label,
            )
            true_cls = labels.round().squeeze().long()

            image_names.extend(filenames)
            true_1.extend(true_cls.numpy().tolist())
            pred_1.extend(pred_cls.numpy().tolist())

    true_1 = np.array(true_1, dtype=int)
    pred_1 = np.array(pred_1, dtype=int)
    true_0 = true_1 - 1
    pred_0 = pred_1 - 1

    return image_names, true_1, pred_1, true_0, pred_0, pred_raw


def main():
    parser = argparse.ArgumentParser(
        description="Compute confusion matrix and accuracy on TEST set"
    )

    parser.add_argument('--run_name',
                help = 'Define the run_name for saving everything (example: test_resnet)',
                default='DummyName')

    parser.add_argument(
        "--param_jsonpath",
        default="config/params_example_ordinal.json",
        help="Path to parameter JSON used for training",
    )
    parser.add_argument(
        "--datapath",
        required=True,
        help="Path to data directory (train/test dirs)",
    )
    parser.add_argument(
        "--savepath",
        default="savedmodel",
        help="Base save path (used to set experiment paths)",
    )

    parser.add_argument('--pretrained_path',
                        default = 'config/pretrained_model.ckpt')

    parser.add_argument(
        "--cv_fold",
        type=int,
        default=0,
        help="Cross-validation fold index",
    )

    args = parser.parse_args()

    training_root_folder = '/sc/home/akshay.gudi/code/CleverHansRegression/each_class_30_Oct/class2/exp1/gpupro/Fold0_DR_ec_30_Oct_2/'
    # sub folder under training_root_folder where trained model is stored
    model_path_sub_folder = 'saved_models/Epoch_50_after_protopushing.pth'
    model_path = training_root_folder + model_path_sub_folder
    
    conf_matrix_dir = "confusion_test"
    output_dir = Path(training_root_folder + conf_matrix_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)
    params_dict = load_json(args_dict['param_jsonpath'])
    params_exp = { **params_dict,**args_dict}
    params = Parameters.from_dict(params_exp)

    params.cv_fold = 1
    params.set_savepaths()

    datamodule = MyDataModuleDiabRet(params)
    test_loader = datamodule.test_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_names, true_1, pred_1, true_0, pred_0, pred_raw = load_model_and_predict(
        str(model_path), test_loader, params, device=device
    )

    # Accuracy
    acc = accuracy_score(true_0, pred_0)
    print(f"Test accuracy: {acc:.4f}")

    # Confusion matrix (computed with 0-based, displayed as 1-based)
    n_classes = 5
    cm = confusion_matrix(true_0, pred_0, labels=np.arange(n_classes))
    print("Confusion matrix (rows=true, cols=pred), classes 1..5:")
    print(cm)

    # CSV: image_name, true_label, predicted_label (1-based), prediction_before_clamp
    df = pd.DataFrame(
        {
            "image_name": [Path(n).name for n in image_names],
            "true_label": true_1,
            "predicted_label": pred_1,
            "prediction_before_clamp": pred_raw,
        }
    )
    csv_path = output_dir / "predictions_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    # Save confusion matrix as CSV (classes 1..5)
    class_names = [str(i) for i in range(1, n_classes + 1)]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / "confusion_matrix_test.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to {cm_csv_path}")

    # Plot confusion matrix
    plot_path = output_dir / "confusion_matrix_test.png"
    plot_confmatrix(cm, class_names, plot_path)
    print(f"Saved confusion plot to {plot_path}")


if __name__ == "__main__":
    main()