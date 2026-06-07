# INSightR-Net

## Requirements

Install dependencies as described in `requirements.txt`:

```bash
conda create -n new_insight_env python=3.10
conda activate new_insight_env
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

All commands below assume:

```bash
cd /path/to/CleverHansRegression
conda activate new_insight_env
```

---

## Trained models

Four INSightR-Net checkpoints (epoch 50, after protopushing) are available from [GitHub Releases](https://github.com/AkshayGudi/CleverHansRegression/releases/tag/v1.0-trained-models):

**[trained_models_DR_checkpoints.zip](https://github.com/AkshayGudi/CleverHansRegression/releases/download/v1.0-trained-models/trained_models_DR_checkpoints.zip)**

### Download and extract

From the **repository root** (`CleverHansRegression/`):

```bash
cd /path/to/CleverHansRegression

wget -O trained_models_DR_checkpoints.zip \
  "https://github.com/AkshayGudi/CleverHansRegression/releases/download/v1.0-trained-models/trained_models_DR_checkpoints.zip"

unzip trained_models_DR_checkpoints.zip
rm trained_models_DR_checkpoints.zip   # optional
```

The zip already contains a top-level `trained_models/` folder, so extracting in the repo root gives:

```
CleverHansRegression/
  trained_models/
    DR-100-fixed/
      Epoch_50_after_protopushing.pth
    DR-100-random/
      Epoch_50_after_protopushing.pth
    DR-50-fixed/
      Epoch_50_after_protopushing.pth
    DR-50-random/
      Epoch_50_after_protopushing.pth
```

| Checkpoint folder | Experiment | Use with `--model_path` / `--ckpt` |
|-------------------|------------|-------------------------------------|
| `trained_models/DR-100-fixed/` | 100% contamination, fixed placement | Matches `data/DR-100-fixed` after data prep |
| `trained_models/DR-100-random/` | 100% contamination, random placement | Matches `data/DR-100-random` |
| `trained_models/DR-50-fixed/` | 50% contamination, fixed placement | Matches `data/DR-50-fixed` |
| `trained_models/DR-50-random/` | 50% contamination, random placement | Matches `data/DR-50-random` |

Pick the checkpoint that matches your prepared data experiment. Example:

```bash
export EXP=DR-100-fixed
export MODEL=trained_models/$EXP/Epoch_50_after_protopushing.pth

python -m metrics.evaluate_ordinal_regression_test \
  --model_path "$MODEL" \
  --datapath data/$EXP \
  --param_jsonpath config/params_example_ordinal.json \
  --test_config config/datasplit/dr_config/dr_test_config.json \
  --focus_class 3 \
  --output_dir outputs/$EXP/metrics
```

All evaluation commands in this readme use paths of the form `trained_models/<experiment>/Epoch_50_after_protopushing.pth`.

Pretrained weights for training from scratch (optional): `config/pretrained_model.ckpt` (see [GitHub Releases](https://github.com/AkshayGudi/CleverHansRegression/releases)).

---

## Data preparation

This repository uses the **same balanced EyePACS subset as the [INSightR-Net paper](https://arxiv.org/abs/2208.00457)** (8,908 train + 6,030 test images), not the full Kaggle release. **Do not upload or redistribute** fundus images.

All preparation scripts and configs are under `data_preparation/` (gitignored image folders: `original_data/`, `data/`).

### Overview

```
Kaggle download  →  copy_insightr_subset.py  →  run_artifact_overlay.py  →  data/<experiment>/
     (external)         original_data/              one folder per experiment
```

| Step | Script | Output location |
|------|--------|-----------------|
| 1. Download | [Kaggle DR competition](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) | Your machine (outside the repo) |
| 2. Subset copy | `data_preparation/copy_insightr_subset.py` | `original_data/original_train/`, `original_data/original_test/` |
| 3. Artifact overlay | `data_preparation/run_artifact_overlay.py` | `data/DR-100-fixed/`, `data/DR-100-random/`, … |

Train/val/test **splits** for evaluation are fixed in `config/datasplit/dr_config/` (`dr_train_config.json`, `dr_test_config.json`) and do not need to be regenerated.

### Step 1 — Download from Kaggle

1. Create a [Kaggle](https://www.kaggle.com/) account and accept the competition rules.
2. Download the [Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) dataset.
3. Extract so you have `train/` and `test/` JPEG folders:

```
/path/to/diabetic-retinopathy-detection/
  train/          # ~35k JPEGs (full Kaggle train set)
  test/           # ~53k JPEGs (full Kaggle test set)
```

CLI example (requires [Kaggle API](https://github.com/Kaggle/kaggle-api) configured):

```bash
kaggle competitions download -c diabetic-retinopathy-detection
unzip train.zip
unzip test.zip
```

### Step 2 — Copy the INSightR-Net subset

Image lists: `data_preparation/DR_train_data.csv` (8,908) and `data_preparation/DR_test_data.csv` (6,030).

From the repository root:

```bash
python data_preparation/copy_insightr_subset.py \
  --kaggle_dir /path/to/diabetic-retinopathy-detection \
  --output_dir original_data
```

Result (repo root):

```
original_data/
  original_train/    # 8,908 JPEGs copied from Kaggle train/
  original_test/     # 6,030 JPEGs copied from Kaggle test/
```

Optional dry run: add `--dry_run`.

### Step 3 — Overlay artifacts (Clever-Hans experiments)

Configs per experiment: `data_preparation/DR-100-fixed/`, `DR-100-random/`, `DR-50-fixed/`, `DR-50-random/`.  
Each folder contains `data_info.json`, `train_labeled_data.csv`, `test_labeled_data.csv`, and `artifact_pos_train.csv` / `artifact_pos_test.csv` (placement source of truth).  
Artifact patch: `data_preparation/art16.png`.

Run once per experiment (from the repository root):

```bash
for exp in DR-100-fixed DR-100-random DR-50-fixed DR-50-random; do
  python data_preparation/run_artifact_overlay.py \
    --experiment "$exp" \
    --original_data_dir original_data \
    --output_dir "data/$exp"
done
```

Single experiment example:

```bash
python data_preparation/run_artifact_overlay.py \
  --experiment DR-100-fixed \
  --original_data_dir original_data \
  --output_dir data/DR-100-fixed
```

| Experiment | Config folder | Placement | Contamination |
|------------|---------------|-----------|---------------|
| `DR-100-fixed` | `data_preparation/DR-100-fixed` | fixed | 100% of class-3 train/test |
| `DR-100-random` | `data_preparation/DR-100-random` | random | 100% |
| `DR-50-fixed` | `data_preparation/DR-50-fixed` | fixed | 50% |
| `DR-50-random` | `data_preparation/DR-50-random` | random | 50% |

If your fundus images are preprocessed elsewhere (e.g. resized/cropped before overlay), pass explicit input dirs:

```bash
python data_preparation/run_artifact_overlay.py \
  --experiment DR-100-fixed \
  --train_input_dir /path/to/preprocessed/train \
  --test_input_dir /path/to/preprocessed/test \
  --output_dir data/DR-100-fixed
```

### Final directory layout

After Step 3, each experiment under `data/` looks like this:

```
data/DR-100-fixed/
  train/                                      # clean + artifact train JPEGs
  test/                                       # clean test JPEGs (no artifact)
  artifact/data_details_class3/
    train_labeled_data.csv                    # image_name, artifact_label (0/1)
    test_labeled_data.csv
    artifact_pos_train.csv                    # center_x, center_y, target_size, …
    artifact_pos_test.csv
    data_info.json
    DR_train_data_patch.csv                   # patch manifests (reference)
    DR_test_data_patch.csv
    train_yellow_patch.csv
    test_yellow_patch.csv
```

The same structure is created for `DR-100-random`, `DR-50-fixed`, and `DR-50-random`.

### Using prepared data in downstream commands

Pick the experiment that matches your checkpoint (`trained_models/<experiment>/`).  
All examples below use **`DR-100-fixed`**; replace with `DR-100-random`, `DR-50-fixed`, or `DR-50-random` as needed.

| Role | Path | Used by |
|------|------|---------|
| Experiment root | `data/DR-100-fixed` | `--datapath` (metrics, training, last-layer retrain) |
| Clean test images | `data/DR-100-fixed/test` | Localization GT masks (`--original_dir`), single-image PRP/PLRP demos |
| Artifact test images | `data/DR-100-fixed/artifact/data_details_class3/test` | Ablation (`--test_dir`), relevance ordering (`--image_dir`), localization (`--artifact_dir`) |
| Artifact labels | `data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv` | Ablation, relevance ordering, localization (`--artifact_csv`) |
| Artifact positions | `data/DR-100-fixed/artifact/data_details_class3/artifact_pos_test.csv` | Localization metrics (`--position_csv`) |
| Train/val/test split | `config/datasplit/dr_config/dr_train_config.json`, `dr_test_config.json` | Metrics, ablation, training (`--test_config`, `--train_split`, `--test_split`) |
| Model checkpoint | `trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth` | All evaluation scripts (`--model_path` / `--ckpt`) |
| Hyperparameters | `config/params_example_ordinal.json` | All scripts (`--param_jsonpath`) |

**INSightR-Net metrics & ablation** — `--datapath data/DR-100-fixed` points at the experiment root (`train/` + `test/`). Ablation and artifact-only evaluation additionally need the artifact test folder and CSVs under `artifact/data_details_class3/`.

**Baseline PRP** — heatmaps use the checkpoint only; relevance ordering and localization need `artifact/data_details_class3/test` plus the labeled/position CSVs. Localization compares clean `test/` vs artifact `artifact/.../test/`.

**Pruned PRP (PLRP)** — same data paths as baseline PRP for localization and paper-metric comparisons (`--original_dir`, `--artifact_dir`, `--artifact_csv`, `--position_csv`).

Quick reference for `DR-100-fixed`:

```bash
export EXP=DR-100-fixed
export DATAPATH=data/$EXP
export MODEL=trained_models/$EXP/Epoch_50_after_protopushing.pth
export ARTIFACT_DIR=$DATAPATH/artifact/data_details_class3
export PARAMS=config/params_example_ordinal.json
export TEST_CONFIG=config/datasplit/dr_config/dr_test_config.json
```

Then, for example:

```bash
# Metrics
python -m metrics.evaluate_ordinal_regression_test \
  --model_path "$MODEL" --datapath "$DATAPATH" --param_jsonpath "$PARAMS" \
  --test_config "$TEST_CONFIG" --focus_class 3 --output_dir outputs/$EXP/metrics

# PRP localization
python -m prp.localization_metrics \
  --ckpt "$MODEL" --param_jsonpath "$PARAMS" \
  --original_dir "$DATAPATH/test" --artifact_dir "$ARTIFACT_DIR/test" \
  --artifact_csv "$ARTIFACT_DIR/test_labeled_data.csv" \
  --position_csv "$ARTIFACT_DIR/artifact_pos_test.csv" \
  --target_class 3 --prototypes 22 24 25 27 28 \
  --num_images 50 --output_dir outputs/$EXP/localization_metrics
```

---

## 1. Baseline INSightR-Net — metrics, visualization, ablation

### 1.1 Ordinal regression test metrics

Confusion matrix, MAE/MSE/RMSE, R², accuracy, quadratic weighted κ (full test set + class-3 subset):

```bash
python -m metrics.evaluate_ordinal_regression_test \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --datapath data/DR-100-fixed \
  --param_jsonpath config/params_example_ordinal.json \
  --test_config config/datasplit/dr_config/dr_test_config.json \
  --focus_class 3 \
  --output_dir outputs/DR-100-fixed/metrics
```

### 1.2 Red activation map visualization

Three-panel figures (original | red overlay | prototype patch) from checkpoint buffers:

```bash
python prototype_visualization/visualize_prototypes_red.py \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --output_dir outputs/DR-100-fixed/red_activation \
  --prototypes 22 24 25 27 28
```

Two-panel variant (for thesis figures):

```bash
python prototype_visualization/visualize_prototypes_red_2panel.py \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --output_dir outputs/DR-100-fixed/red_activation_2panel \
  --prototypes 22 24 25 27 28
```

### 1.3 Prototype ablation

**Single-prototype sweep (no retraining)** — accuracy/MAE drop when each prototype is masked:

```bash
python -m ablation.run_prototype_ablation \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --test_dir data/DR-100-fixed/artifact/data_details_class3/test \
  --test_config config/datasplit/dr_config/dr_test_config.json \
  --artifact_csv data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv \
  --target_class 3 \
  --ablation_indices 22 24 25 27 28 \
  --output_dir outputs/DR-100-fixed/ablation \
  --device cuda
```

**Multi-prototype ablation (no retraining)** — remove several prototypes at once:

```bash
python -m ablation.ablation_multiple \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --test_dir data/DR-100-fixed/artifact/data_details_class3/test \
  --test_config config/datasplit/dr_config/dr_test_config.json \
  --artifact_csv data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv \
  --target_class 3 \
  --artifact_only \
  --prototypes_to_remove 24 25 27 \
  --output_dir outputs/DR-100-fixed/ablation_multiple \
  --device cuda
```

**Last-layer retrain after prototype removal** — remove prototypes, freeze the rest of the network, retrain only the last layer, then re-evaluate:

```bash
python -m ablation.last_layer_retrain \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --datapath data/DR-100-fixed \
  --train_split config/datasplit/dr_config/dr_train_config.json \
  --test_split config/datasplit/dr_config/dr_test_config.json \
  --prototypes_to_remove 24 25 27 28 \
  --epochs 5 \
  --lr 1e-3 \
  --batch_size 30 \
  --output_dir outputs/DR-100-fixed/last_layer_retrain \
  --device cuda
```

---

## 2. Baseline PRP (Prototypical Relevance Propagation)

### 2.1 Generate PRP heatmaps

Stored prototype images (from checkpoint):

```bash
python prp/main_generate_prp.py \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --output_dir outputs/DR-100-fixed/prp \
  --prototypes 22 24 25 27 28
```

PRP on a single test image:

```bash
python prp/main_generate_prp.py \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --output_dir outputs/DR-100-fixed/prp_test_image \
  --test_image_path data/DR-100-fixed/test/12345_left.jpeg \
  --prototypes 22 24 25
```

### 2.2 Relevance ordering (insertion test)

Artifact-stratified relevance ordering (PRP vs prototype heatmap vs random):

```bash
python prp/relevance_ordering_proto.py \
  --artifact_labels_csv data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv \
  --image_dir data/DR-100-fixed/artifact/data_details_class3/test \
  --output_dir outputs/DR-100-fixed/relevance_ordering \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --prototypes 22 24 25 27 28 \
  --num_random_images 50 \
  --seed 42
```

### 2.3 Localization metrics (PRP vs activation)

Pointing Game, RMA, RRA, ROC AUC, Top-K IoU (GT mask from clean vs artifact image diff):

```bash
python -m prp.localization_metrics \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --original_dir data/DR-100-fixed/test \
  --artifact_dir data/DR-100-fixed/artifact/data_details_class3/test \
  --artifact_csv data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv \
  --position_csv data/DR-100-fixed/artifact/data_details_class3/artifact_pos_test.csv \
  --target_class 3 \
  --prototypes 22 24 25 27 28 \
  --num_images 50 \
  --output_dir outputs/DR-100-fixed/localization_metrics
```

Self-check (no data):

```bash
python -m prp.localization_metrics --self_check
```

---

## 3. Pruned PRP (PLRP-λ)

Default pruning: `plrp_p_pos=0.25`, `plrp_p_neg=0.125`.  
Sanity check: set both to `0.0` (must match baseline PRP).

```bash
python -m plrp_ext.tests.test_plrp_equivalence \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --prototypes 22 24 25
```

### 3.1 Generate PLRP-PRP heatmaps

```bash
python plrp_ext/main_generate_plrp.py \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --output_dir outputs/DR-100-fixed/plrp \
  --prototypes 22 24 25 27 28 \
  --plrp_p_pos 0.25 \
  --plrp_p_neg 0.125
```

### 3.2 Four-panel comparison (activation | baseline PRP | PLRP-PRP)

```bash
python plrp_ext/compare_insightr_prp_plrp.py \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --prototypes 22 24 25 27 28 \
  --plrp_p_pos 0.25 \
  --plrp_p_neg 0.125 \
  --output_dir outputs/DR-100-fixed/compare_insightr_prp_plrp
```

On a test image:

```bash
python plrp_ext/compare_insightr_prp_plrp.py \
  --model_path trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --test_image_path data/DR-100-fixed/test/12345_left.jpeg \
  --prototypes 22 24 25 \
  --plrp_p_pos 0.25 \
  --plrp_p_neg 0.125 \
  --output_dir outputs/DR-100-fixed/compare_insightr_prp_plrp_testimg
```

### 3.3 PRP vs PLRP localization comparison

```bash
python plrp_ext/compare_prp_plrp_localization.py \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --original_dir data/DR-100-fixed/test \
  --artifact_dir data/DR-100-fixed/artifact/data_details_class3/test \
  --artifact_csv data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv \
  --position_csv data/DR-100-fixed/artifact/data_details_class3/artifact_pos_test.csv \
  --prototypes 22 24 25 27 28 \
  --num_images 50 \
  --plrp_p_pos 0.25 \
  --plrp_p_neg 0.125 \
  --output_dir outputs/DR-100-fixed/compare_prp_plrp_localization
```

### 3.4 PRP vs PLRP paper metrics (sparsity & perturbation faithfulness)

```bash
python plrp_ext/compare_prp_plrp_paper_metrics.py \
  --ckpt trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth \
  --param_jsonpath config/params_example_ordinal.json \
  --original_dir data/DR-100-fixed/test \
  --artifact_dir data/DR-100-fixed/artifact/data_details_class3/test \
  --artifact_csv data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv \
  --prototypes 22 24 25 27 28 \
  --num_images 50 \
  --plrp_p_pos 0.25 \
  --plrp_p_neg 0.125 \
  --output_dir outputs/DR-100-fixed/compare_prp_plrp_paper_metrics
```

---

## Training (optional)

INSightR-Net:

```bash
python main_trainer.py \
  --param_jsonpath config/params_example_ordinal.json \
  --datapath data/DR-100-fixed \
  --savepath outputs/DR-100-fixed/training \
  --pretrained_path config/pretrained_model.ckpt
```

Baseline model:

```bash
python main_trainer_baseline.py \
  --param_jsonpath config/params_example_baseline.json \
  --datapath data/DR-100-fixed \
  --savepath outputs/DR-100-fixed/baseline_training \
  --pretrained_path config/pretrained_model.ckpt
```

Parameters are defined in `define_parameters.py` and overridden via JSON (see `config/params_example_ordinal.json`).

## Logging

Training logs to MLflow. View with:

```bash
mlflow ui
```

run from the repository root.
