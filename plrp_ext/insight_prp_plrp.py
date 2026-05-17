"""
Prototypical Relevance Propagation with Pruned LRP (PLRP-PRP) for INSightR-Net.

Direct copy of ../insight_prp.py with imports rewired to the PLRP-extended
LRP utilities in lrp_general6_plrp.py. The custom L2 prototype rule
(``l2_lrp_insightr``) is intentionally identical to the parent file: PLRP is
applied only to the standard backbone rules (Conv2d beta=0 and Linear epsilon),
following the scope defined in Yanez Sarmiento et al., 2024 (Section 3).

When PLRP is disabled (p_pos == p_neg == 0, the default), this file produces
heatmaps that are bit-identical to the parent insight_prp.py. This invariant
is verified by ``tests/test_plrp_equivalence.py``.

Imports below explicitly target the local PLRP-extended modules. The parent
module (``insight_prp.py``) is never imported, so it is impossible for code in
this file to alter the behaviour of the standard PRP pipeline.
"""

from __future__ import print_function, division

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from insight_training.resnet_features import BasicBlock, Bottleneck, ResNet_features
from plrp_ext.lrp_general6_plrp import (
    sum_stacked2,
    get_lrpwrapperformodule,
    resetbn,
    bnafterconv_overwrite_intoconv,
    relu_wrapper_fct,
    sigmoid_wrapper_fct,
    conv2d_beta0_wrapper_fct,
    linearlayer_eps_wrapper_fct,
    adaptiveavgpool2d_wrapper_fct,
    maxpool2d_wrapper_fct,
    eltwisesum_stacked2_eps_wrapper_fct,
    safe_divide,
)


# ──────────────────────────────────────────────────────────
# LRP default parameters and method mapping
# ──────────────────────────────────────────────────────────

LRP_PARAMS = {
    'conv2d_ignorebias': True,
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': True,
    'lowest': torch.tensor(0.0),
    'highest': torch.tensor(1.0),
}

LRP_LAYER2METHOD = {
    'nn.ReLU': relu_wrapper_fct,
    'nn.Sigmoid': sigmoid_wrapper_fct,
    'nn.BatchNorm2d': relu_wrapper_fct,
    'nn.Conv2d': conv2d_beta0_wrapper_fct,
    'nn.Linear': linearlayer_eps_wrapper_fct,
    'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
    'nn.MaxPool2d': maxpool2d_wrapper_fct,
    'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct,
}


# ──────────────────────────────────────────────────────────
# Canonized ResNet blocks (residual as explicit sum_stacked2)
# ──────────────────────────────────────────────────────────

class BasicBlock_fused(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample)
        self.elt = sum_stacked2()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.elt(torch.stack([out, identity], dim=0))
        out = self.relu(out)
        return out


class Bottleneck_fused(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample)
        self.elt = sum_stacked2()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.elt(torch.stack([out, identity], dim=0))
        out = self.relu(out)
        return out


# ──────────────────────────────────────────────────────────
# Canonized ResNet feature extractor
# ──────────────────────────────────────────────────────────

class ResNet_canonized(ResNet_features):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)

    def setbyname(self, name, value):
        def iteratset(obj, components, value):
            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                return True
            else:
                return iteratset(getattr(obj, components[0]), components[1:], value)
        components = name.split('.')
        return iteratset(self, components, value)

    def copyfromresnet(self, net, lrp_params, lrp_layer2method):
        """Copy weights from a trained ResNet, fusing BatchNorm into Conv and wrapping all layers."""
        updated_layers_names = []
        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            if isinstance(src_module, nn.Linear):
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise RuntimeError(f"Could not find module {src_module_name} in target net")
                updated_layers_names.append(src_module_name)

            if isinstance(src_module, nn.Conv2d):
                last_src_module_name = src_module_name
                last_src_module = src_module

            if isinstance(src_module, nn.BatchNorm2d):
                thisis_inputconv = (lrp_params['use_zbeta'] and last_src_module_name == 'conv1')
                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv)
                if not self.setbyname(last_src_module_name, wrapped):
                    raise RuntimeError(f"Could not find module {last_src_module_name} in target net")
                updated_layers_names.append(last_src_module_name)

                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise RuntimeError(f"Could not find module {src_module_name} in target net")
                updated_layers_names.append(src_module_name)

        for target_module_name, target_module in self.named_modules():
            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise RuntimeError(f"Could not find module {target_module_name} in target net")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise RuntimeError(f"Could not find module {target_module_name} in target net")
                updated_layers_names.append(target_module_name)


# ──────────────────────────────────────────────────────────
# Canonized add-on layers (matches INSightR-Net architecture)
# ──────────────────────────────────────────────────────────

class AddonCanonized(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.addon = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )


# ──────────────────────────────────────────────────────────
# Architecture lookup for canonized ResNets
# ──────────────────────────────────────────────────────────

def _resnet_canonized(block, layers, **kwargs):
    return ResNet_canonized(block, layers, **kwargs)

_base_arch_to_canonized = {
    'resnet18': lambda **kw: _resnet_canonized(BasicBlock_fused, [2, 2, 2, 2], **kw),
    'resnet34': lambda **kw: _resnet_canonized(BasicBlock_fused, [3, 4, 6, 3], **kw),
    'resnet50': lambda **kw: _resnet_canonized(Bottleneck_fused, [3, 4, 6, 3], **kw),
    'resnet101': lambda **kw: _resnet_canonized(Bottleneck_fused, [3, 4, 23, 3], **kw),
    'resnet152': lambda **kw: _resnet_canonized(Bottleneck_fused, [3, 8, 36, 3], **kw),
}


# ──────────────────────────────────────────────────────────
# Custom L2 LRP for INSightR-Net prototype similarity
# (intentionally NOT pruned -- not a standard LRP rule)
# ──────────────────────────────────────────────────────────

class l2_lrp_insightr(torch.autograd.Function):
    """
    Forward: computes L2 distances and converts to similarities.
    Backward: distributes relevance across channels based on per-channel L2 proximity.
    """

    @staticmethod
    def forward(ctx, conv_features, model):
        ctx.save_for_backward(conv_features, model.prototype_vectors)

        x2 = conv_features ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=model.ones)

        p2 = model.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=conv_features, weight=model.prototype_vectors)
        intermediate_result = -2 * xp + p2_reshape
        distances = F.relu(x2_patch_sum + intermediate_result)

        proto_activation = getattr(model, 'proto_activation', 'log')
        epsilon = getattr(model, 'epsilon', 1e-4)

        if proto_activation == 'log':
            similarities = torch.log((distances + 1) / (distances + epsilon))
        elif proto_activation == 'exp_norm':
            proto_shape = model.proto_shape
            dist_max = proto_shape[1] * proto_shape[2] * proto_shape[3]
            similarities = 1.0 / ((distances / dist_max) + epsilon)
        elif proto_activation == 'linear':
            similarities = -distances
        else:
            similarities = torch.log((distances + 1) / (distances + epsilon))

        return similarities

    @staticmethod
    def backward(ctx, grad_output):
        """Distributes relevance based on inverse squared per-channel L2 distance."""
        conv, prototypes = ctx.saved_tensors
        i = conv.shape[2]
        j = conv.shape[3]
        c = conv.shape[1]
        p = prototypes.shape[0]

        conv_expanded = conv.repeat(p, 1, 1, 1)
        prototype_expanded = prototypes.repeat(1, 1, i, j)

        conv_expanded = conv_expanded.squeeze()

        l2 = conv_expanded - prototype_expanded
        d = 1.0 / (l2 ** 2 + 1e-12)

        denom = torch.sum(d, dim=1, keepdim=True) + 1e-12
        denom = denom.repeat(1, c, 1, 1) + 1e-12
        R = torch.div(d, denom)

        grad_output = grad_output.repeat(c, 1, 1, 1)
        grad_output = grad_output.permute(1, 0, 2, 3)

        R = R * grad_output
        R = torch.sum(R, dim=0)
        R = torch.unsqueeze(R, dim=0)

        return R, None


# ──────────────────────────────────────────────────────────
# Utility: set module attribute by dotted name
# ──────────────────────────────────────────────────────────

def _setbyname(obj, name, value):
    def iteratset(obj, components, value):
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            return iteratset(getattr(obj, components[0]), components[1:], value)
    components = name.split('.')
    return iteratset(obj, components, value)


# ──────────────────────────────────────────────────────────
# Main canonization function: wrap INSightR-Net for PLRP-PRP
# ──────────────────────────────────────────────────────────

def PRPCanonizedModel(ppnet, lrp_params=None, lrp_layer2method=None):
    """
    Wraps a trained INSightR-Net (ppnet) with LRP-aware layers for PLRP-PRP.

    Identical to the parent insight_prp.PRPCanonizedModel except that the
    wrappers come from lrp_general6_plrp (which apply PLRP-lambda pruning at
    Conv2d beta=0 and Linear epsilon backwards when set_plrp_params has been
    called with non-zero proportions).
    """
    if lrp_params is None:
        lrp_params = LRP_PARAMS
    if lrp_layer2method is None:
        lrp_layer2method = LRP_LAYER2METHOD

    device = next(ppnet.parameters()).device
    base_arch = getattr(ppnet, 'base_architecture', None)

    if base_arch is None:
        features_repr = str(ppnet.features).upper()
        if 'RESNET_FEATURES' in features_repr:
            block = ppnet.features.block
            layers = ppnet.features.layers
            if block == BasicBlock:
                if layers == [2, 2, 2, 2]:
                    base_arch = 'resnet18'
                elif layers == [3, 4, 6, 3]:
                    base_arch = 'resnet34'
            elif block == Bottleneck:
                if layers == [3, 4, 6, 3]:
                    base_arch = 'resnet50'
                elif layers == [3, 4, 23, 3]:
                    base_arch = 'resnet101'
                elif layers == [3, 8, 36, 3]:
                    base_arch = 'resnet152'

    if base_arch is None:
        raise ValueError("Could not determine base architecture. Pass it explicitly via ppnet.base_architecture.")

    if base_arch not in _base_arch_to_canonized:
        raise ValueError(f"Unsupported architecture: {base_arch}. Supported: {list(_base_arch_to_canonized.keys())}")

    canonized_backbone = _base_arch_to_canonized[base_arch]()
    canonized_backbone = canonized_backbone.to(device)
    canonized_backbone.copyfromresnet(ppnet.features, lrp_params=lrp_params, lrp_layer2method=lrp_layer2method)
    canonized_backbone = canonized_backbone.to(device)
    ppnet.features = canonized_backbone

    proto_shape = ppnet.proto_shape
    first_add_on = ppnet.add_on_layers[0]
    in_channels = first_add_on.module.in_channels if hasattr(first_add_on, 'module') else first_add_on.in_channels
    stride = first_add_on.module.stride[0] if hasattr(first_add_on, 'module') else first_add_on.stride[0]

    addon_canonized = AddonCanonized(in_channels=in_channels, out_channels=proto_shape[1], stride=stride)
    for src_module_name, src_module in ppnet.add_on_layers.named_modules():
        if isinstance(src_module, nn.Conv2d):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
            _setbyname(addon_canonized.addon, src_module_name, wrapped)
        if isinstance(src_module, nn.ReLU):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
            _setbyname(addon_canonized.addon, src_module_name, wrapped)
        if isinstance(src_module, nn.Sigmoid):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
            _setbyname(addon_canonized.addon, src_module_name, wrapped)

    addon_canonized = addon_canonized.to(device)
    ppnet.add_on_layers = addon_canonized.addon

    spatial_size = ppnet.output_size_conv
    ppnet.max_layer = nn.MaxPool2d((spatial_size, spatial_size), return_indices=False)
    ppnet.max_layer = get_lrpwrapperformodule(copy.deepcopy(ppnet.max_layer), lrp_params, lrp_layer2method)

    return ppnet


# ──────────────────────────────────────────────────────────
# PRP heatmap generation
# ──────────────────────────────────────────────────────────

def compute_heatmap(relevance, percentile=100):
    """Convert raw relevance tensor to a normalized 2D heatmap."""
    hm = relevance.squeeze().sum(dim=0).detach().cpu().numpy()
    clim = np.percentile(np.abs(hm), percentile)
    if clim > 0:
        hm = hm / clim
    return hm


def generate_prp_image(inputs, pno, model, device):
    """Generate a PLRP-PRP heatmap for a single image and a single prototype."""
    model.train(False)
    inputs = inputs.to(device).clone()
    inputs.requires_grad = True

    with torch.enable_grad():
        conv_features = model.conv_features(inputs)
        similarities = l2_lrp_insightr.apply(conv_features, model)
        max_similarities = model.max_layer(similarities)
        max_similarities = max_similarities.view(-1, model.num_prototypes)

    max_similarities[:, pno].backward()

    rel = inputs.grad.data
    heatmap = compute_heatmap(rel.to('cpu'))
    return heatmap


def generate_prp_all_prototypes(model, device, output_dir, comparison_fn=None, percentile=100):
    """Generate PLRP-PRP heatmaps for all prototypes using their stored source images."""
    os.makedirs(output_dir, exist_ok=True)
    model.train(False)

    num_prototypes = model.num_prototypes
    heatmaps = {}

    for pno in range(num_prototypes):
        proto_img = model.prototype_images[pno]

        if proto_img.max() == 0:
            print(f"Prototype {pno}: no stored image (all zeros), skipping.")
            continue

        proto_dir = os.path.join(output_dir, f"prototype_{pno}")
        os.makedirs(proto_dir, exist_ok=True)

        img_float = proto_img.float() / 255.0
        img_tensor = img_float.permute(2, 0, 1).unsqueeze(0)

        heatmap = generate_prp_image(img_tensor, pno, model, device)
        heatmaps[pno] = heatmap

        plt.imsave(os.path.join(proto_dir, "heatmap.png"), heatmap, cmap="seismic", vmin=-1, vmax=1)

        orig_img_np = proto_img.cpu().numpy() / 255.0
        plt.imsave(os.path.join(proto_dir, "original.png"), orig_img_np)

        prp_overlay = _create_overlay(orig_img_np, heatmap)
        plt.imsave(os.path.join(proto_dir, "overlay.png"), prp_overlay, vmin=0, vmax=1)

        if comparison_fn is not None:
            comparison_fn(orig_img_np, prp_overlay, pno, proto_dir)

        print(f"Prototype {pno}/{num_prototypes - 1}: saved to {proto_dir}/")

    return heatmaps


def generate_prp_for_image(
    image_tensor,
    model,
    device,
    output_dir,
    prototype_indices=None,
    raw_image_np=None,
):
    """Generate PLRP-PRP heatmaps for a test image across specified (or all) prototypes."""
    os.makedirs(output_dir, exist_ok=True)
    model.train(False)

    if prototype_indices is None:
        prototype_indices = list(range(model.num_prototypes))

    heatmaps = {}
    for pno in prototype_indices:
        heatmap = generate_prp_image(image_tensor, pno, model, device)
        heatmaps[pno] = heatmap

        plt.imsave(
            os.path.join(output_dir, f"prp_testimage_proto_{pno}.png"),
            heatmap, cmap="seismic", vmin=-1, vmax=1,
        )

        if raw_image_np is not None:
            overlay = _create_overlay(raw_image_np, heatmap)
            plt.imsave(
                os.path.join(output_dir, f"prp_testimage_proto_{pno}_overlay.png"),
                overlay, vmin=0, vmax=1,
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(raw_image_np)
            axes[0].set_title('Test Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            axes[1].imshow(overlay)
            axes[1].set_title('PLRP-PRP Relevance Overlay', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            axes[2].imshow(heatmap, cmap='seismic', vmin=-1, vmax=1)
            axes[2].set_title('Raw PLRP-PRP Heatmap', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            fig.suptitle(
                f'Prototype {pno} -- Test-image PLRP-PRP',
                fontsize=16, fontweight='bold', y=0.98,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            plt.savefig(
                os.path.join(output_dir, f"prp_testimage_proto_{pno}_comparison.png"),
                dpi=150, bbox_inches='tight',
            )
            plt.close(fig)

    return heatmaps


# ──────────────────────────────────────────────────────────
# Overlay utility
# ──────────────────────────────────────────────────────────

def _create_overlay(original_img, heatmap, max_alpha=0.85):
    """Blend a PLRP-PRP heatmap onto the original image."""
    import cv2
    h, w = original_img.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_clipped = np.clip(hm_resized, -1, 1)

    intensity = np.sqrt(np.abs(hm_clipped))
    alpha = (intensity * max_alpha)[:, :, np.newaxis]

    hm_color = np.zeros((h, w, 3), dtype=np.float32)
    pos_mask = hm_clipped > 0
    neg_mask = hm_clipped < 0

    pos_strength = np.sqrt(hm_clipped[pos_mask])
    hm_color[pos_mask, 0] = 1.0
    hm_color[pos_mask, 1] = pos_strength * 0.4
    hm_color[pos_mask, 2] = 0

    hm_color[neg_mask, 0] = 0
    hm_color[neg_mask, 1] = 0
    hm_color[neg_mask, 2] = 1.0

    dim_factor = 1.0 - 0.3 * (1.0 - intensity[:, :, np.newaxis])
    dimmed_original = original_img * dim_factor

    overlay = (1 - alpha) * dimmed_original + alpha * hm_color
    overlay = np.clip(overlay, 0, 1)
    return overlay
