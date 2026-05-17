# `plrp_ext/` — Pruned LRP for INSightR-Net's PRP

This folder is a **strictly additive** extension of the existing PRP code in
`CleverHansRegression/`. It implements **PLRP-λ** (Yanez Sarmiento et al.,
*Sparse Explanations of Neural Networks Using Pruned Layer-Wise Relevance
Propagation*, ECML PKDD 2024) on top of the parent PRP pipeline.

> **Isolation guarantee.** No file outside this folder is modified, renamed,
> or moved. `main_generate_prp.py`, `insight_prp.py`, `lrp_general6.py`,
> the evaluation scripts, the model code, and the training pipeline are all
> untouched and continue to behave exactly as before. Even if everything in
> this folder breaks, the standard PRP pipeline keeps working.

The implementation deliberately covers the **smallest, defensible scope**
that is still genuinely PLRP rather than post-hoc thresholding:

* **PLRP-λ only** (the simpler of the two paper variants; see paper §3.2.2).
  No PLRP-M.
* **Fixed-proportion mode only** (a single `(p_pos, p_neg)` per run).
  No sparsity-gain mode.
* Pruning is applied **only at the standard parametric LRP rules**, namely
  `Conv2d` (β=0) and `Linear` (ε). The first-layer z-β conv, pooling layers,
  ReLU, Sigmoid, the residual eltwise sum, and the **L2 prototype rule of
  PRP** are all left mathematically unchanged. This matches the PLRP paper's
  scope (Convolution + Linear) and the explicit "first layer is left
  unpruned" statement in §3.2.

When `p_pos == p_neg == 0` the pipeline behaves **identically** to the
standard PRP code by construction (early return inside the pruning helper).
This invariant is verified by `tests/test_plrp_equivalence.py`.

## Files

| File | Mirrors | Differences from parent |
|------|---------|--------------------------|
| `lrp_general6_plrp.py` | `../lrp_general6.py` | adds `set_plrp_params`, `_prune_relevance_lambda`, applies pruning at end of `conv2d_beta0_wrapper_fct.backward` and `linearlayer_eps_wrapper_fct.backward` |
| `insight_prp_plrp.py` | `../insight_prp.py` | imports come from `lrp_general6_plrp` instead of `lrp_general6`; `l2_lrp_insightr` is unchanged |
| `main_generate_plrp.py` | `../main_generate_prp.py` | adds `--plrp_p_pos` and `--plrp_p_neg` CLI flags; figure titles say "PLRP-PRP" |
| `tests/test_plrp_equivalence.py` | (new) | sanity check: with `p=0`, PLRP-PRP heatmap == standard PRP heatmap |
| `compare_prp_plrp_localization.py` | (new) | **Quantitative** PRP vs PLRP-PRP on the same artifact images (reuses `evaluation.localization_metrics.all_metrics`) |

## Quantitative comparison: PRP vs PLRP-PRP (localization)

`compare_prp_plrp_localization.py` runs **baseline PRP** (`insight_prp`) and
**PLRP-PRP** (`plrp_ext` + `set_plrp_params`) on the **same** fundus images and
GT artifact masks as `evaluation/localization_metrics.py`, then writes:

* `per_image_metrics.csv` — all methods per image and prototype
* `summary_per_prototype.csv` — mean / std per prototype
* `paired_deltas_per_image.csv` — per (image, prototype), `metric_plrp − metric_prp`
* `comparison_paired.json` — overall mean delta and **fraction of pairs where PLRP is strictly better** per metric

Use the same `--artifact_dir`, `--artifact_csv`, and (recommended)
`--position_csv` as for `python -m evaluation.localization_metrics`. Example:

```bash
cd /sc/home/akshay.gudi/code/CleverHansRegression
conda activate new_insight_env

python3 plrp_ext/compare_prp_plrp_localization.py \
    --ckpt bld_art_25_Apr/class3_v14_2_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_03_May_3/saved_models/Epoch_50_after_protopushing.pth \
    --param_jsonpath config/params_example_ordinal.json \
    --artifact_dir /path/to/your/bld_artifact/test/jpeg \
    --artifact_csv /path/to/test_labeled_data.csv \
    --position_csv /path/to/artifact_pos_test.csv \
    --prototypes 22 27 33 \
    --num_images 50 \
    --plrp_p_pos 0.25 \
    --plrp_p_neg 0.125 \
    --output_dir plrp_ext/outputs/loc_compare_class3_v14_2

# Faster: skip prototype_activation column
python3 plrp_ext/compare_prp_plrp_localization.py ... --skip_prototype_activation
```

**How to read `comparison_paired.json`:** for each metric, higher is better.
If `mean_delta_plrp_minus_prp` is positive and
`fraction_strictly_positive` is well above 0.5, PLRP improves localization
on average for your setup. If deltas are mixed or negative, report that
honestly (still a valid thesis result).

## How to run

All commands assume your usual CleverHansRegression conda environment. Run
from the **project root** (`/sc/home/akshay.gudi/code/CleverHansRegression`).
The script `plrp_ext/main_generate_plrp.py` prepends the repo root to
`sys.path` automatically, so ``python3 plrp_ext/main_generate_plrp.py``
works even though Python would otherwise only put ``plrp_ext/`` on the path
(which caused ``ModuleNotFoundError: No module named 'helpers'`` before
that fix).

```bash
cd /sc/home/akshay.gudi/code/CleverHansRegression
conda activate new_insight_env
```

### Step 0 — sanity check (mandatory before any results run)

This must pass before you trust any PLRP outputs:

```bash
python3 -m plrp_ext.tests.test_plrp_equivalence \
    --model_path=/sc/home/akshay.gudi/code/CleverHansRegression/bld_art_25_Apr/class3_v4_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_25_Apr_3/saved_models/Epoch_50_after_protopushing.pth \
    --param_jsonpath=config/params_example_ordinal.json \
    --prototypes 0 5 16 22 33
```

Expected last line: `PASS: PLRP pipeline with p=0 reproduces unpruned PRP within tolerance.`
Tolerances: `max_abs<=1e-4`, `mean_abs<=1e-5`, relative-L1 `<=1e-4`. If
this fails, **do not proceed**: the pruning implementation has a bug that
must be fixed first.

You can also run the same check on a real test image (in case stored
prototype images differ in content from the test set):

```bash
python3 -m plrp_ext.tests.test_plrp_equivalence \
    --model_path=/path/to/Epoch_50_after_protopushing.pth \
    --param_jsonpath=config/params_example_ordinal.json \
    --test_image_path=/path/to/some_test_image.jpeg \
    --prototypes 0 5 16
```

### Step 1 — generate a PLRP-PRP heatmap on stored prototype images

Mirrors `main_generate_prp.py` exactly, with two extra flags. Output goes to a
**separate** directory so it never overwrites your existing PRP results:

```bash
python3 plrp_ext/main_generate_plrp.py \
    --model_path=/path/to/Epoch_50_after_protopushing.pth \
    --param_jsonpath=config/params_example_ordinal.json \
    --output_dir=plrp_ext/outputs/p_pos_025/class3_v4 \
    --prototypes 2 8 11 16 22 23 24 25 26 27 31 33 37 38 39 \
    --plrp_p_pos 0.25 \
    --plrp_p_neg 0.125
```

Setting `--plrp_p_pos 0.0 --plrp_p_neg 0.0` makes the script reproduce the
unpruned PRP output (useful as a per-image debug check). The script prints
which mode it is in:

```
PLRP pruning DISABLED (p_pos=0, p_neg=0). Output should match unpruned PRP exactly.
```
or
```
PLRP-lambda ENABLED with p_pos=0.25, p_neg=0.125.
```

### Step 2 — generate a PLRP-PRP heatmap on a test image

```bash
python3 plrp_ext/main_generate_plrp.py \
    --model_path=/path/to/Epoch_50_after_protopushing.pth \
    --param_jsonpath=config/params_example_ordinal.json \
    --output_dir=plrp_ext/outputs/p_pos_025/test_image_artifact \
    --test_image_path=/path/to/artifact_test_image.jpeg \
    --prototypes 16 22 33 \
    --plrp_p_pos 0.25 \
    --plrp_p_neg 0.125
```

### Step 3 — sweep `p` for the localization-metric study

The PLRP paper finds that Relevance Mass Accuracy increases with `p` up to
a point and then decreases. To find the operating point for your
Clever-Hans benchmark, run the same images at multiple `p` values:

```bash
for p in 0.0 0.10 0.25 0.40 0.50; do
    python3 plrp_ext/main_generate_plrp.py \
        --model_path=/path/to/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --output_dir=plrp_ext/outputs/sweep/p_${p} \
        --prototypes 16 22 33 \
        --plrp_p_pos ${p} \
        --plrp_p_neg $(echo "${p} / 2" | bc -l)
done
```

(The paper uses `p_neg = p_pos / 2` as a default; adjust to taste.)

Then point your existing localization-metric runner
(`evaluation/localization_metrics.py`) at each output directory and tabulate
Pointing Game / RMA / RRA / ROC-AUC / Top-K IoU per `p`. The existing runner
is unchanged and reads PRP-format heatmap outputs, which is what this folder
produces.

## What changes vs. parent PRP, in one sentence per file

* `lrp_general6_plrp.py`: at the end of the conv-β0 and linear-ε backwards,
  `R = _maybe_prune(R)` is added. `_maybe_prune` is a no-op when PLRP is
  disabled (`p=0`), and otherwise applies PLRP-λ per-sample as defined in
  Eqs. (3.2) of the paper.
* `insight_prp_plrp.py`: imports `lrp_general6_plrp` instead of
  `lrp_general6`. The L2 prototype rule (`l2_lrp_insightr`) is identical to
  the parent.
* `main_generate_plrp.py`: same CLI as the parent plus
  `--plrp_p_pos`/`--plrp_p_neg`; calls `set_plrp_params` before
  canonization.

## What this implementation does NOT do (out of scope, by design)

* **PLRP-M.** Requires a two-pass backward (compute unpruned R, then re-run
  with masked activations). Not implemented; would need a re-architecture
  of the custom-autograd backwards.
* **Sparsity-gain threshold mode.** Requires solving the PLRP sparsity-gain
  inequality per layer per input. Not implemented; only fixed-proportion
  mode is exposed.
* **PLRP at the L2 prototype rule.** PLRP is defined for standard LRP
  rules; whether and how to extend it to the channel-wise inverse-squared
  redistribution of the L2 layer is an open research question. The L2 rule
  here is identical to the parent PRP code.
* **Modifications outside this folder.** Nothing else is touched.

These three items are correctly placed in the *outlook / future work*
section of the thesis.

## Citation guidance for the thesis

When you write up the PLRP integration, the contribution claim should be
something like:

> We integrate Pruned Layer-wise Relevance Propagation (PLRP-λ; Yanez
> Sarmiento et al., 2024) into the PRP backbone of our adapted INSightR-Net
> pipeline. PLRP is applied to the standard LRP rules (Conv2d β=0 and
> Linear ε) of the canonized ResNet; the prototype-specific L2 rule of
> Gautam et al. (2022) is kept unchanged, following the scope defined in
> the PLRP paper. We compare PLRP-PRP against unpruned PRP on our synthetic
> Clever-Hans diabetic-retinopathy benchmark using localization metrics
> (Pointing Game, Relevance Mass Accuracy, Relevance Rank Accuracy,
> ROC-AUC, Top-K IoU) and a relevance-ordering / insertion curve.
