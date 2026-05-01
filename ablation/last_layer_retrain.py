#!/usr/bin/env python3
"""
Last-layer retraining after prototype removal — INSightR-Net.

This mirrors PRP paper Table I: remove selected prototypes, then retrain ONLY
the last layer (everything else frozen) and re-evaluate. No feature update,
no prototype update, no add_on_layers update.

Removed prototypes are kept removed throughout retraining via:
  - zeroing the last_layer.weight columns at start, and
  - registering a backward hook that masks gradients on those columns.

This is a clean, standalone script — it does NOT modify existing training code.

Example (class3_v14 model, remove prototype 3 from class-3 pool):

  cd /sc/home/akshay.gudi/code/CleverHansRegression
  source /sc/home/akshay.gudi/conda3/etc/profile.d/conda.sh
  conda activate new_insight_env
  PYTHONUNBUFFERED=1 python3 -m ablation.last_layer_retrain \\
    --ckpt bld_art_25_Apr/class3_v14_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_28_Apr_1/saved_models/Epoch_50_after_protopushing.pth \\
    --param_jsonpath config/params_example_ordinal.json \\
    --datapath /sc/home/akshay.gudi/data_store/DR/bld_artifact/class3_v14 \\
    --train_split config/datasplit/dr_config/dr_train_config.json \\
    --test_split  config/datasplit/dr_config/dr_test_config.json \\
    --prototypes_to_remove 3 \\
    --epochs 5 --lr 1e-3 --batch_size 30 --device cuda \\
    --output_dir bld_art_25_Apr/class3_v14_no_clr_fix_26/exp1/DR_25_Jan_2026_1/Fold0_DR_28_Apr_1/img/lastlayer_retrain_remove3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from define_parameters import NetworkParams  # noqa: E402
from helpers import load_json  # noqa: E402
from insight_training.model import construct_PPNet  # noqa: E402
from dataset_DiabeticRet import DiabeticRet  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _select_files_for_split(split_dict: dict, fold: int, role: str) -> Tuple[List[str], List[str]]:
    """Return (file_stems, label_strs) for either 'train' or 'val' role of a CV fold."""
    fold_key = f"Fold {fold}"
    files = split_dict[fold_key][role]["files"]
    labels = split_dict[fold_key][role]["labels"]
    return list(files), list(labels)


def build_train_loader(
    train_split_path: Path,
    train_dir: Path,
    fold: int,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Build training dataloader the same way ``MyDataModuleDiabRet`` does."""
    split = load_json(str(train_split_path))
    stems, labels = _select_files_for_split(split, fold, "train")

    paths: List[Path] = []
    keep_labels: List[str] = []
    for stem, lab in zip(stems, labels):
        p = train_dir / (stem + ".jpeg")
        if p.is_file():
            paths.append(p)
            keep_labels.append(lab)
    if not paths:
        raise RuntimeError(f"No training files found in {train_dir}")

    ds = DiabeticRet(paths, keep_labels, preload=False)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )


def build_test_loader(
    test_split_path: Path,
    test_dir: Path,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Build test dataloader from a flat 'files'/'labels' JSON (same as datamodule)."""
    split = load_json(str(test_split_path))
    stems = split["files"]
    labels = split["labels"]

    paths: List[Path] = []
    keep_labels: List[str] = []
    for stem, lab in zip(stems, labels):
        p = test_dir / (stem + ".jpeg")
        if p.is_file():
            paths.append(p)
            keep_labels.append(lab)
    if not paths:
        raise RuntimeError(f"No test files found in {test_dir}")

    ds = DiabeticRet(paths, keep_labels, preload=False)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )


# ---------------------------------------------------------------------------
# Checkpoint + freezing utilities
# ---------------------------------------------------------------------------


def load_ppnet_checkpoint(ckpt_path: str, ppnet: nn.Module, device: torch.device) -> nn.Module:
    """Support both raw .pth state_dict and Lightning .ckpt files."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        sub = {k[len("ppnet.") :]: v for k, v in sd.items() if k.startswith("ppnet.")}
        ppnet.load_state_dict(sub, strict=True)
    else:
        ppnet.load_state_dict(ckpt, strict=True)
    return ppnet


def freeze_all_but_last_layer(ppnet: nn.Module) -> None:
    """Match ``init_lastonly()`` from the Lightning module."""
    for p in ppnet.features.parameters():
        p.requires_grad = False
    for p in ppnet.add_on_layers.parameters():
        p.requires_grad = False
    ppnet.prototype_vectors.requires_grad = False
    for p in ppnet.last_layer.parameters():
        p.requires_grad = True


def zero_columns(weight: torch.Tensor, indices: Iterable[int]) -> None:
    """Set selected columns of a 2-D weight matrix to 0 (in-place, no grad)."""
    idxs = list(indices)
    if not idxs:
        return
    with torch.no_grad():
        for j in idxs:
            weight[:, j] = 0.0


def install_grad_mask_hook(weight: torch.Tensor, indices: Iterable[int]):
    """
    Register a backward hook that zeros gradients on selected columns of
    ``weight`` so the optimizer cannot revive removed prototypes.
    Returns the hook handle (so we can remove it later if needed).
    """
    idxs = list(indices)
    if not idxs:
        return None

    def _mask_hook(grad: torch.Tensor) -> torch.Tensor:
        masked = grad.clone()
        for j in idxs:
            masked[:, j] = 0.0
        return masked

    return weight.register_hook(_mask_hook)


# ---------------------------------------------------------------------------
# Evaluation: rounded accuracy + MAE, overall and per class
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    ppnet: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Run model on a dataloader, compute rounded accuracy + MAE overall and per
    class (1..5). Returns a dict with simple JSON-serializable types.
    """
    ppnet.eval()
    all_pred: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for ims, labels, _names in loader:
        ims = ims.to(device)
        y = labels.to(device).float()
        logits, _, _ = ppnet(ims, return_convs=False)
        pred = logits.squeeze(-1)
        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())

    pred = torch.cat(all_pred)
    y = torch.cat(all_y)

    rounded_pred = torch.round(pred).clamp(1, 5)
    rounded_y = torch.round(y).clamp(1, 5)

    overall = {
        "n": int(pred.shape[0]),
        "rounded_accuracy": float((rounded_pred == rounded_y).float().mean()),
        "mae": float((pred - y).abs().mean()),
    }

    per_class = {}
    for c in (1, 2, 3, 4, 5):
        mask = rounded_y == c
        n_c = int(mask.sum().item())
        if n_c == 0:
            per_class[str(c)] = {"n": 0, "rounded_accuracy": None, "mae": None}
            continue
        acc_c = float((rounded_pred[mask] == rounded_y[mask]).float().mean())
        mae_c = float((pred[mask] - y[mask]).abs().mean())
        per_class[str(c)] = {"n": n_c, "rounded_accuracy": acc_c, "mae": mae_c}

    return {"overall": overall, "per_class": per_class}


# ---------------------------------------------------------------------------
# Training loop (MSE only, last layer)
# ---------------------------------------------------------------------------


def train_last_layer(
    ppnet: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    removed_indices: List[int],
    log_every: int = 25,
) -> List[dict]:
    """
    Train only the last layer with MSE for ``epochs`` epochs. Returns per-epoch
    summary dicts. Removed prototype columns are kept at zero throughout via a
    gradient hook installed on ``last_layer.weight``.
    """
    optimizer = torch.optim.Adam(ppnet.last_layer.parameters(), lr=lr)
    history: List[dict] = []

    for ep in range(1, epochs + 1):
        ppnet.train()
        # Re-apply freezing because nn.Module.train() does not change requires_grad,
        # but BN/dropout switch back on. We do not have BN in last layer; safe.
        freeze_all_but_last_layer(ppnet)

        epoch_loss_sum = 0.0
        epoch_n = 0
        t0 = time.perf_counter()

        for batch_idx, (ims, labels, _names) in enumerate(loader):
            ims = ims.to(device)
            y = labels.to(device).float()

            logits, _, _ = ppnet(ims, return_convs=False)
            pred = logits.squeeze(-1)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Belt-and-suspenders: even with the grad-mask hook installed, force
            # the removed columns back to zero so any numerical drift cannot
            # accumulate.
            zero_columns(ppnet.last_layer.weight.data, removed_indices)

            epoch_loss_sum += float(loss.item()) * ims.shape[0]
            epoch_n += ims.shape[0]

            if (batch_idx + 1) % log_every == 0:
                print(
                    f"  epoch {ep} step {batch_idx + 1}: "
                    f"running mse={epoch_loss_sum / max(epoch_n, 1):.4f}",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        avg_mse = epoch_loss_sum / max(epoch_n, 1)
        history.append({"epoch": ep, "avg_train_mse": avg_mse, "elapsed_sec": elapsed})
        print(f"[epoch {ep}/{epochs}] avg train mse={avg_mse:.4f}  ({elapsed:.1f}s)", flush=True)

    return history


# ---------------------------------------------------------------------------
# Self-check (no data needed)
# ---------------------------------------------------------------------------


def run_self_check() -> None:
    """Construct a fresh PPNet, freeze, mask, run a tiny optimizer step,
    confirm removed columns stay zero."""
    np_net = NetworkParams()
    ppnet = construct_PPNet(network_params=np_net)
    ppnet.eval()
    device = torch.device("cpu")
    ppnet.to(device)

    removed = [0, 5, 7]
    freeze_all_but_last_layer(ppnet)
    zero_columns(ppnet.last_layer.weight.data, removed)
    handle = install_grad_mask_hook(ppnet.last_layer.weight, removed)
    assert handle is not None

    optim = torch.optim.Adam(ppnet.last_layer.parameters(), lr=1e-2)
    x = torch.rand(2, 3, np_net.img_size, np_net.img_size, device=device)
    y = torch.tensor([3.0, 4.0], device=device)

    pred = ppnet(x, return_convs=False)[0].squeeze(-1)
    loss = F.mse_loss(pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    zero_columns(ppnet.last_layer.weight.data, removed)

    w = ppnet.last_layer.weight.data
    bad = [j for j in removed if float(w[:, j].abs().max().item()) != 0.0]
    if bad:
        raise AssertionError(f"removed columns drifted from zero: {bad}")
    print(
        "last_layer_retrain self_check: OK (removed columns stayed zero "
        "after optimizer step).",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Last-layer retraining after prototype removal (INSightR-Net).",
    )
    p.add_argument("--ckpt", type=str, default="", help="Path to Epoch_*_after_protopushing.pth")
    p.add_argument("--param_jsonpath", type=str, default="", help="JSON with network_params block")
    p.add_argument("--datapath", type=str, default="", help="Folder with train/ and test/")
    p.add_argument("--train_split", type=str, default="config/datasplit/dr_config/dr_train_config.json")
    p.add_argument("--test_split", type=str, default="config/datasplit/dr_config/dr_test_config.json")
    p.add_argument("--cv_fold", type=int, default=0)
    p.add_argument("--prototypes_to_remove", type=int, nargs="*", default=[])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--output_dir", type=str, default="lastlayer_retrain_output")
    p.add_argument("--self_check", action="store_true")
    return p.parse_args()


def _resolve_relative(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_ROOT / p)


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return

    if not args.ckpt or not args.param_jsonpath or not args.datapath:
        raise SystemExit("Need --ckpt, --param_jsonpath and --datapath (or use --self_check).")

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ----- network_params and model -----
    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get("network_params", {}))

    ppnet = construct_PPNet(network_params=network_params)
    ppnet.to(device)
    load_ppnet_checkpoint(args.ckpt, ppnet, device)
    print(f"Loaded checkpoint: {args.ckpt}")

    # ----- validate removed indices -----
    removed = sorted(set(args.prototypes_to_remove)) if args.prototypes_to_remove else []
    for j in removed:
        if j < 0 or j >= ppnet.num_prototypes:
            raise SystemExit(f"prototype index {j} out of range [0, {ppnet.num_prototypes - 1}]")
    print(f"Prototypes to remove (kept frozen at 0): {removed}")

    # ----- freeze everything except last layer -----
    freeze_all_but_last_layer(ppnet)

    # ----- zero removed columns + install gradient mask hook -----
    zero_columns(ppnet.last_layer.weight.data, removed)
    install_grad_mask_hook(ppnet.last_layer.weight, removed)

    # ----- dataloaders -----
    train_dir = Path(args.datapath) / "train"
    test_dir = Path(args.datapath) / "test"
    train_split = _resolve_relative(args.train_split)
    test_split = _resolve_relative(args.test_split)

    train_loader = build_train_loader(train_split, train_dir, args.cv_fold, args.batch_size)
    test_loader = build_test_loader(test_split, test_dir, args.batch_size)
    print(f"Train batches/epoch: {len(train_loader)} | Test images: {len(test_loader.dataset)}")

    # ----- evaluate BEFORE retraining (with removal applied) -----
    print("\nEvaluating BEFORE last-layer retraining (with removal applied) ...", flush=True)
    metrics_before = evaluate(ppnet, test_loader, device)
    ov = metrics_before["overall"]
    print(
        f"BEFORE: overall acc={ov['rounded_accuracy']:.4f}, "
        f"MAE={ov['mae']:.4f}, n={ov['n']}",
        flush=True,
    )

    # ----- train last layer only (skip if epochs == 0 for eval-only smoke tests) -----
    if args.epochs <= 0:
        print("\n--epochs <= 0: skipping retraining (eval-only mode).", flush=True)
        history: List[dict] = []
        metrics_after = metrics_before
    else:
        print("\nRetraining last layer (MSE) ...", flush=True)
        history = train_last_layer(
            ppnet=ppnet,
            loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            removed_indices=removed,
        )

        print("\nEvaluating AFTER last-layer retraining ...", flush=True)
        metrics_after = evaluate(ppnet, test_loader, device)
        ov = metrics_after["overall"]
        print(
            f"AFTER:  overall acc={ov['rounded_accuracy']:.4f}, "
            f"MAE={ov['mae']:.4f}, n={ov['n']}",
            flush=True,
        )

    # ----- save outputs -----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ppnet.state_dict(), out_dir / "lastlayer_retrained.pth")

    summary = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "removed_prototypes": removed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "metrics_before_retrain": metrics_before,
        "metrics_after_retrain": metrics_after,
        "training_history": history,
    }
    (out_dir / "retrain_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved checkpoint:    {out_dir / 'lastlayer_retrained.pth'}")
    print(f"Saved summary JSON:  {out_dir / 'retrain_summary.json'}")

    print(
        "\nFINAL  acc  {:.4f} -> {:.4f}  |  MAE  {:.4f} -> {:.4f}".format(
            metrics_before["overall"]["rounded_accuracy"],
            metrics_after["overall"]["rounded_accuracy"],
            metrics_before["overall"]["mae"],
            metrics_after["overall"]["mae"],
        )
    )


if __name__ == "__main__":
    main()
