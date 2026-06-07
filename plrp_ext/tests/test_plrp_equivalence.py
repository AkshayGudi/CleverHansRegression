"""
Sanity test: PLRP-PRP with p_pos=p_neg=0 must equal unpruned PRP.

This script loads a trained INSightR-Net checkpoint, generates a heatmap
once via the existing parent PRP pipeline (insight_prp.py +
lrp_general6.py), and once via the new PLRP-extended pipeline
(plrp_ext.insight_prp_plrp + plrp_ext.lrp_general6_plrp) with PLRP
disabled, and compares them.

Pass criterion (all must hold):
    - shape(prp_heatmap) == shape(plrp_heatmap)
    - max abs diff < REL_TOL_MAX_ABS
    - mean abs diff < REL_TOL_MEAN_ABS
    - relative L1 diff < REL_TOL_REL_L1

If this test FAILS, do NOT trust any PLRP results from this code base.
Investigate before proceeding.

Usage (from the CleverHansRegression repository root):

    conda activate new_insight_env

    python3 -m plrp_ext.tests.test_plrp_equivalence \
        --model_path=/path/to/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --prototypes 0 5 16

You can also point at a test image instead of using stored prototype
images:

    python3 -m plrp_ext.tests.test_plrp_equivalence \
        --model_path=/path/to/Epoch_50_after_protopushing.pth \
        --param_jsonpath=config/params_example_ordinal.json \
        --test_image_path=/path/to/test_image.jpeg \
        --prototypes 0 5 16
"""

import argparse
import sys
from pathlib import Path

# Same as main_generate_plrp.py: ensure repo root is on sys.path when this
# file is executed directly (not only via ``python -m``).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import cv2
from pathlib import Path

from helpers import load_json
from define_parameters import NetworkParams
from insight_training.model import construct_PPNet

# Existing (unmodified) PRP pipeline
import prp.insight_prp as parent_insight_prp

# New PLRP-extended pipeline
from plrp_ext.lrp_general6_plrp import set_plrp_params
from plrp_ext import insight_prp_plrp


REL_TOL_MAX_ABS = 1e-4
REL_TOL_MEAN_ABS = 1e-5
REL_TOL_REL_L1 = 1e-4


def _load_ppnet(model_path, network_params, device):
    """Load a trained INSightR-Net (PPNet) from .pth or .ckpt."""
    checkpoint = torch.load(model_path, map_location=device)
    ppnet = construct_PPNet(network_params=network_params)
    ppnet = ppnet.to(device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        ppnet_state_dict = {
            key[len('ppnet.'):]: value
            for key, value in state_dict.items()
            if key.startswith('ppnet.')
        }
        ppnet.load_state_dict(ppnet_state_dict, strict=True)
    else:
        ppnet.load_state_dict(checkpoint, strict=True)
    ppnet.eval()
    return ppnet


def _stored_prototype_image_tensor(prp_model, pno):
    """Returns (image_tensor (1,3,H,W), or None if prototype is empty)."""
    proto_img = prp_model.prototype_images[pno]
    if proto_img.max() == 0:
        return None
    img_float = proto_img.float() / 255.0
    img_tensor = img_float.permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def _load_test_image(image_path, img_size):
    """Load a test image and return the model-input tensor (1,3,H,W)."""
    jpeg_im = cv2.imread(str(image_path))
    if jpeg_im is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if jpeg_im.shape[0] != img_size or jpeg_im.shape[1] != img_size:
        jpeg_im = cv2.resize(jpeg_im, (img_size, img_size))
    norm = jpeg_im / 255.0
    return torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()


def _build_models(model_path, network_params, device):
    """Returns (parent_prp_model, plrp_prp_model). Both share the same trained weights."""
    ppnet_a = _load_ppnet(model_path, network_params, device)
    ppnet_a.base_architecture = network_params.base_architecture
    parent_prp_model = parent_insight_prp.PRPCanonizedModel(ppnet_a).to(device)

    ppnet_b = _load_ppnet(model_path, network_params, device)
    ppnet_b.base_architecture = network_params.base_architecture
    plrp_prp_model = insight_prp_plrp.PRPCanonizedModel(ppnet_b).to(device)

    return parent_prp_model, plrp_prp_model


def _compare(hm_parent, hm_plrp):
    """Compute diff statistics between two heatmaps."""
    if hm_parent.shape != hm_plrp.shape:
        return {
            'ok': False,
            'reason': f'shape mismatch: parent={hm_parent.shape}, plrp={hm_plrp.shape}',
            'max_abs': None, 'mean_abs': None, 'rel_l1': None,
        }
    diff = hm_plrp - hm_parent
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    parent_l1 = float(np.sum(np.abs(hm_parent)))
    rel_l1 = float(np.sum(np.abs(diff))) / max(parent_l1, 1e-12)
    ok = (
        max_abs <= REL_TOL_MAX_ABS
        and mean_abs <= REL_TOL_MEAN_ABS
        and rel_l1 <= REL_TOL_REL_L1
    )
    reason = '' if ok else (
        f'tolerances exceeded: max_abs={max_abs:.3e} (lim {REL_TOL_MAX_ABS:.0e}), '
        f'mean_abs={mean_abs:.3e} (lim {REL_TOL_MEAN_ABS:.0e}), '
        f'rel_l1={rel_l1:.3e} (lim {REL_TOL_REL_L1:.0e})'
    )
    return {'ok': ok, 'reason': reason,
            'max_abs': max_abs, 'mean_abs': mean_abs, 'rel_l1': rel_l1}


def main():
    parser = argparse.ArgumentParser(
        description='Sanity test: PLRP-PRP with p=0 must match unpruned PRP.'
    )
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--param_jsonpath', required=True)
    parser.add_argument('--prototypes', type=int, nargs='+', default=[0],
                        help='Prototype indices to compare (default: 0).')
    parser.add_argument('--test_image_path', type=str, default=None,
                        help='Optional: compare on this test image instead of stored prototype images.')
    args = parser.parse_args()

    set_plrp_params(p_pos=0.0, p_neg=0.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    params_dict = load_json(args.param_jsonpath)
    network_params = NetworkParams.from_dict(params_dict.get('network_params', {}))
    print(f'Network: {network_params.base_architecture}, '
          f'proto_shape={network_params.proto_shape}, '
          f'img_size={network_params.img_size}')

    print('Building parent PRP and PLRP-PRP models from the same checkpoint...')
    parent_model, plrp_model = _build_models(args.model_path, network_params, device)

    print(f'Comparing {len(args.prototypes)} prototype(s): {args.prototypes}')
    print(f'Tolerances: max_abs<={REL_TOL_MAX_ABS:.0e}, '
          f'mean_abs<={REL_TOL_MEAN_ABS:.0e}, rel_l1<={REL_TOL_REL_L1:.0e}')

    results = []
    for pno in args.prototypes:
        if args.test_image_path is not None:
            img_tensor = _load_test_image(args.test_image_path, network_params.img_size)
            tag = f'test_image proto={pno}'
        else:
            img_tensor = _stored_prototype_image_tensor(parent_model, pno)
            if img_tensor is None:
                print(f'  proto {pno}: stored prototype image is empty -- SKIPPED')
                continue
            tag = f'stored proto={pno}'

        hm_parent = parent_insight_prp.generate_prp_image(
            img_tensor.clone(), pno, parent_model, device
        )
        hm_plrp = insight_prp_plrp.generate_prp_image(
            img_tensor.clone(), pno, plrp_model, device
        )
        cmp = _compare(hm_parent, hm_plrp)
        results.append((tag, cmp))
        status = 'OK' if cmp['ok'] else 'FAIL'
        print(f'  {tag}: {status}  '
              f'(max_abs={cmp["max_abs"]:.3e}, '
              f'mean_abs={cmp["mean_abs"]:.3e}, '
              f'rel_l1={cmp["rel_l1"]:.3e})')
        if not cmp['ok']:
            print(f'    REASON: {cmp["reason"]}')

    n_total = len(results)
    n_ok = sum(1 for _, c in results if c['ok'])
    print()
    print(f'Result: {n_ok}/{n_total} prototype(s) within tolerance.')
    if n_ok != n_total:
        print('FAIL: PLRP pipeline with p=0 does NOT match unpruned PRP. Investigate before running PLRP experiments.')
        sys.exit(1)
    print('PASS: PLRP pipeline with p=0 reproduces unpruned PRP within tolerance.')


if __name__ == '__main__':
    main()
