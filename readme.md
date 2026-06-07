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

Download the four released checkpoints and extract them under `trained_models/`:

```
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

Pretrained weights for training (optional): `config/pretrained_model.ckpt` (see GitHub Releases).

---

## Data

### Download

Fundus images: [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) (EyePACS; restricted license — do not redistribute preprocessed images).

### Preprocess and layout

1. Extract EyePACS train/test images.
2. Preprocess fundus images (Ben Graham-style resize/normalization).  
   *(Preprocessing script instructions will be added here.)*
3. Place preprocessed data in per-experiment folders:

```
data/
  DR-100-fixed/
    train/
    test/
  DR-100-random/
    train/
    test/
  DR-50-fixed/
    train/
    test/
  DR-50-random/
    train/
    test/
```

Train/val/test splits are defined in `config/datasplit/` (see `config/datasplit/dr_config/`).

### Artifact overlay

Synthetic artifact overlays use the scripts and assets in `artifact/` (including `art16.png`).  
For each experiment, run the overlay pipeline and keep outputs under:

```
data/<experiment>/artifact/data_details_class3/
  test_labeled_data.csv      # columns: image_name, artifact_label
  artifact_pos_test.csv      # artifact centre positions (for localization metrics)
  test/                      # artifact-added test JPEGs
```

Clean (no-artifact) test images remain in `data/<experiment>/test/`.

**Example paths used below (DR-100-fixed):**

| Variable | Path |
|----------|------|
| `MODEL` | `trained_models/DR-100-fixed/Epoch_50_after_protopushing.pth` |
| `DATAPATH` | `data/DR-100-fixed` |
| `TEST_DIR` | `data/DR-100-fixed/test` |
| `ARTIFACT_DIR` | `data/DR-100-fixed/artifact/data_details_class3/test` |
| `ARTIFACT_CSV` | `data/DR-100-fixed/artifact/data_details_class3/test_labeled_data.csv` |
| `POSITION_CSV` | `data/DR-100-fixed/artifact/data_details_class3/artifact_pos_test.csv` |
| `TEST_CONFIG` | `config/datasplit/dr_config/dr_test_config.json` |
| `PARAMS` | `config/params_example_ordinal.json` |

Replace `DR-100-fixed` with `DR-100-random`, `DR-50-fixed`, or `DR-50-random` for the other models.

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
