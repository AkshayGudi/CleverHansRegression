"""
Relevance ordering for fixed prototype(s).

Modes
-----
1) **Artifact-stratified (paper-style sampling)**  
   Pass ``--artifact_labels_csv`` plus ``--image_dir``, ``--output_dir``, ``--model_path``,
   ``--param_jsonpath``, and ``--prototype_index`` *or* ``--prototypes``.  
   For each prototype index ``p``, the script samples up to ``--num_random_images`` (default 50)
   rows with ``artifact_label == 1`` and up to 50 with ``artifact_label == 0`` from the CSV,
   then runs the insertion relevance-ordering test (PRP vs upsampled prototype vs random) on
   those lists separately. Results:

     ``<output_dir>/prototype_<p>/with_artifact/*_with_artifact.{csv,png}``
     ``<output_dir>/prototype_<p>/without_artifact/*_without_artifact.{csv,png}``

   CSV columns: ``image_name``, ``artifact_label`` (0/1). Images are read as
   ``<image_dir>/<image_name>.jpeg``.

2) **Checkpoint-only** (no image folder): omit ``--image_dir`` / ``--image_path`` / CSV from the
   subprocess forward args — the wrapper adds ``--from_stored_prototypes`` and calls
   ``relevance_ordering_paper.py`` (uses ``prototype_images[p]`` in the checkpoint).

3) **Forward-only** (legacy): pass ``--image_dir`` (or path/CSV) in the remaining arguments
   without ``--artifact_labels_csv``; wrapper delegates to ``relevance_ordering_paper.py``.

Examples (artifact-stratified):

  python relevance_ordering_proto.py \\
      --artifact_labels_csv /path/to/test_labeled.csv \\
      --image_dir /path/to/jpeg \\
      --output_dir /path/to/ro_results \\
      --model_path /path/to/Epoch_50_after_protopushing.pth \\
      --param_jsonpath /path/to/params.json \\
      --prototype_index 45

  python relevance_ordering_proto.py \\
      --artifact_labels_csv ... --image_dir ... --output_dir ... \\
      --model_path ... --param_jsonpath ... \\
      --prototypes 40 41 45 \\
      --num_random_images 50 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent


def _peek_param_jsonpath(argv: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == "--param_jsonpath" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--param_jsonpath="):
            return a.split("=", 1)[1]
    return None


def _forward_uses_external_images(forward: List[str]) -> bool:
    i = 0
    while i < len(forward):
        a = forward[i]
        if a in ("--image_dir", "--image_path", "--image_list_csv"):
            return True
        if a.startswith("--image_dir=") or a.startswith("--image_path=") or a.startswith(
            "--image_list_csv="
        ):
            return True
        i += 1
    return False


def _validate_indices(indices: List[int], param_jsonpath: Optional[str]) -> None:
    if param_jsonpath is None:
        return
    try:
        from helpers import load_json
        from define_parameters import NetworkParams

        params_dict = load_json(param_jsonpath)
        np_cfg = NetworkParams.from_dict(params_dict.get("network_params", {}))
        n = int(np_cfg.proto_shape[0])
    except Exception as e:
        print(f"Warning: could not validate prototype indices from JSON: {e}", file=sys.stderr)
        return
    bad = [p for p in indices if p < 0 or p >= n]
    if bad:
        raise SystemExit(
            f"Prototype index(es) {bad} out of range; valid [0, {n - 1}] "
            f"(num_prototypes={n})."
        )


def _sample_random_paths(paths: List[Path], n: int, rng: np.random.Generator) -> List[Path]:
    paths = list(paths)
    if len(paths) <= n:
        return paths
    idx = rng.choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def _run_artifact_stratified(args: argparse.Namespace) -> None:
    from define_parameters import NetworkParams
    from helpers import load_json
    from insight_prp import PRPCanonizedModel
    from relevance_ordering_artifact_split import load_image_paths_from_artifact_csv
    from relevance_ordering_paper import load_ppnet, run_relevance_ordering_core

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params_dict = load_json(args.param_jsonpath)
    np_cfg = NetworkParams.from_dict(params_dict.get("network_params", {}))
    img_size = np_cfg.img_size
    num_proto = int(np_cfg.proto_shape[0])

    plist: List[int]
    if args.prototype_index is not None:
        plist = [args.prototype_index]
    else:
        plist = list(dict.fromkeys(args.prototypes))

    for p in plist:
        if p < 0 or p >= num_proto:
            raise SystemExit(f"prototype index {p} out of range; valid 0..{num_proto - 1}")

    ppnet = load_ppnet(args.model_path, np_cfg, device).to(device)
    ppnet_for_prp = load_ppnet(args.model_path, np_cfg, device)
    ppnet_for_prp.base_architecture = np_cfg.base_architecture
    prp_model = PRPCanonizedModel(ppnet_for_prp).to(device)
    print("Canonized model ready for PRP.")

    with_art, without_art = load_image_paths_from_artifact_csv(
        args.artifact_labels_csv, args.image_dir
    )
    rng = np.random.default_rng(args.seed)
    samp_with = _sample_random_paths(with_art, args.num_random_images, rng)
    samp_without = _sample_random_paths(without_art, args.num_random_images, rng)

    print(
        f"Sampled {len(samp_with)} image(s) with artifact (requested up to {args.num_random_images}), "
        f"{len(samp_without)} without artifact."
    )

    fractions = np.linspace(0.0, 1.0, args.num_fractions).tolist()
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    for pno in plist:
        proto_root = base_out / f"prototype_{pno}"
        print(f"\n========== Prototype {pno} → {proto_root} ==========")

        if samp_with:
            sub = proto_root / "with_artifact"
            print(f"  Running with_artifact ({len(samp_with)} images) …")
            run_relevance_ordering_core(
                ppnet,
                prp_model,
                samp_with,
                sub,
                fractions,
                img_size,
                device,
                seed=args.seed,
                fixed_prototype_index=pno,
                topk_prototypes=1,
                prototype_plot_label=pno,
                output_filename_suffix="_with_artifact",
            )
        else:
            print("  [with_artifact] No images — skipped.")

        if samp_without:
            sub = proto_root / "without_artifact"
            print(f"  Running without_artifact ({len(samp_without)} images) …")
            run_relevance_ordering_core(
                ppnet,
                prp_model,
                samp_without,
                sub,
                fractions,
                img_size,
                device,
                seed=args.seed + 10_000,
                fixed_prototype_index=pno,
                topk_prototypes=1,
                prototype_plot_label=pno,
                output_filename_suffix="_without_artifact",
            )
        else:
            print("  [without_artifact] No images — skipped.")

    print(f"\nDone. Results under: {base_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relevance ordering for fixed prototype(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for modes. relevance_ordering_paper.py --help for forwarded options.",
    )
    parser.add_argument(
        "--artifact_labels_csv",
        default=None,
        help="CSV with image_name, artifact_label (1=artifact, 0=clean). Enables stratified 50/50-style runs.",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Folder of .jpeg images (required with --artifact_labels_csv).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root output directory (required with --artifact_labels_csv).",
    )
    parser.add_argument("--model_path", default=None, help="Checkpoint .pth or .ckpt (artifact CSV mode).")
    parser.add_argument("--param_jsonpath", default=None, help="Parameters JSON (artifact CSV mode).")
    parser.add_argument(
        "--num_random_images",
        type=int,
        default=50,
        help="Max images per artifact group to sample (default 50; use all if fewer exist).",
    )
    parser.add_argument("--num_fractions", type=int, default=21, help="Insertion curve resolution.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling.")

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--prototype_index", type=int, help="Single prototype index (0-based).")
    g.add_argument("--prototypes", type=int, nargs="+", help="Several prototype indices.")

    args, forward = parser.parse_known_args()

    dup = [
        x
        for x in forward
        if x in ("--prototype_index", "--prototypes")
        or x.startswith("--prototype_index=")
        or x.startswith("--prototypes=")
    ]
    if dup:
        raise SystemExit("Do not duplicate --prototype_index / --prototypes in forwarded args.")

    pj = args.param_jsonpath or _peek_param_jsonpath(forward)
    if args.prototype_index is not None:
        _validate_indices([args.prototype_index], pj)
    else:
        _validate_indices(list(dict.fromkeys(args.prototypes)), pj)

    if args.artifact_labels_csv:
        missing = [
            n
            for n, v in (
                ("--image_dir", args.image_dir),
                ("--output_dir", args.output_dir),
                ("--model_path", args.model_path),
                ("--param_jsonpath", args.param_jsonpath),
            )
            if not v
        ]
        if missing:
            raise SystemExit(
                f"With --artifact_labels_csv, provide: {', '.join(missing)} "
                "(all of --image_dir --output_dir --model_path --param_jsonpath)."
            )
        _run_artifact_stratified(args)
        return

    proto_extra = (
        ["--prototype_index", str(args.prototype_index)]
        if args.prototype_index is not None
        else ["--prototypes", *[str(x) for x in args.prototypes]]
    )

    paper = _REPO / "relevance_ordering_paper.py"
    if not paper.is_file():
        raise SystemExit(f"Missing {paper}")

    argv = [sys.executable, str(paper)]
    if not _forward_uses_external_images(forward):
        argv.append("--from_stored_prototypes")
    argv.extend(proto_extra)
    argv.extend(forward)

    if any(x == "--from_stored_prototypes" for x in forward):
        raise SystemExit("Do not pass --from_stored_prototypes in forwarded args; the wrapper sets it when appropriate.")

    raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
