"""
Prototype Relevance Propagation (PRP) adapted for INSightR-Net.

Provides:
  - Model canonization: transforms a trained INSightR-Net into an LRP-enabled
    version for generating pixel-level relevance heatmaps.
  - Per-prototype PRP: heatmap showing which pixels activated a specific prototype.
  - Full-prediction PRP: heatmap showing which pixels drove the final regression
    prediction (propagates through the full weighted-mean computation).

Based on: "This looks more like that" (Gautam et al., Pattern Recognition 2022)
Adapted from: https://github.com/SrishtiGautam/PRP
"""

from __future__ import print_function, division

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from insight_training.resnet_features import BasicBlock, Bottleneck, ResNet_features
from insight_training.lrp_general6 import (
    sum_stacked2,
    get_lrpwrapperformodule,
    bnafterconv_overwrite_intoconv,
    resetbn,
    relu_wrapper_fct,
    sigmoid_wrapper_fct,
    conv2d_beta0_wrapper_fct,
    linearlayer_eps_wrapper_fct,
    adaptiveavgpool2d_wrapper_fct,
    maxpool2d_wrapper_fct,
    eltwisesum_stacked2_eps_wrapper_fct,
    safe_divide,
    lrp_backward,
)


# ─────────────────────────────────────────────────────────────────────────────
# Canonized ResNet blocks with explicit element-wise sum for LRP
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock_fused(BasicBlock):
    """BasicBlock with the residual += replaced by an explicit sum_stacked2
    so that LRP can distribute relevance between main path and shortcut."""
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
    """Bottleneck with explicit sum_stacked2 for LRP-compatible skip connections."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Canonized ResNet that can absorb weights from a trained ResNet_features
# ─────────────────────────────────────────────────────────────────────────────

class Modulenotfounderror(Exception):
    pass


class ResNet_canonized(ResNet_features):
    """ResNet_features with all layers replaced by LRP-wrapped versions.
    BatchNorm is fused into the preceding Conv, and skip connections use
    sum_stacked2 for proper relevance splitting."""

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
        """Copy weights from a trained ResNet_features, fuse BatchNorm into Conv,
        and wrap every layer with its LRP-compatible version."""
        updated_layers_names = []
        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            if isinstance(src_module, nn.Linear):
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(f"Could not find module {src_module_name} in target net")
                updated_layers_names.append(src_module_name)

            if isinstance(src_module, nn.Conv2d):
                last_src_module_name = src_module_name
                last_src_module = src_module

            if isinstance(src_module, nn.BatchNorm2d):
                thisis_inputconv_andiwant_zbeta = (
                    lrp_params['use_zbeta'] and last_src_module_name == 'conv1'
                )
                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                wrapped = get_lrpwrapperformodule(
                    m, lrp_params, lrp_layer2method,
                    thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)
                if not self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(f"Could not find module {last_src_module_name} in target net")
                updated_layers_names.append(last_src_module_name)

                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(f"Could not find module {src_module_name} in target net")
                updated_layers_names.append(src_module_name)

        for target_module_name, target_module in self.named_modules():
            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(f"Could not find module {target_module_name} in target net")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(f"Could not find module {target_module_name}")
                updated_layers_names.append(target_module_name)


# ─────────────────────────────────────────────────────────────────────────────
# Canonized add-on layers
# ─────────────────────────────────────────────────────────────────────────────

class addon_canonized(nn.Module):
    """Placeholder add-on layer structure matching INSightR-Net's architecture.
    Weights will be copied from the trained model and wrapped with LRP."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.addon = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Architecture lookup for canonized ResNets
# ─────────────────────────────────────────────────────────────────────────────

def _resnet_canonized(block, layers, **kwargs):
    return ResNet_canonized(block, layers, **kwargs)


def resnet18_canonized(**kwargs):
    return _resnet_canonized(BasicBlock_fused, [2, 2, 2, 2], **kwargs)

def resnet34_canonized(**kwargs):
    return _resnet_canonized(BasicBlock_fused, [3, 4, 6, 3], **kwargs)

def resnet50_canonized(**kwargs):
    return _resnet_canonized(Bottleneck_fused, [3, 4, 6, 3], **kwargs)

def resnet101_canonized(**kwargs):
    return _resnet_canonized(Bottleneck_fused, [3, 4, 23, 3], **kwargs)

def resnet152_canonized(**kwargs):
    return _resnet_canonized(Bottleneck_fused, [3, 8, 36, 3], **kwargs)


base_architecture_to_features = {
    'resnet18': resnet18_canonized,
    'resnet34': resnet34_canonized,
    'resnet50': resnet50_canonized,
    'resnet101': resnet101_canonized,
    'resnet152': resnet152_canonized,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper to set attributes by dotted name
# ─────────────────────────────────────────────────────────────────────────────

def setbyname(obj, name, value):
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


# ─────────────────────────────────────────────────────────────────────────────
# Default LRP parameters
# ─────────────────────────────────────────────────────────────────────────────

LRP_PARAMS = {
    'conv2d_ignorebias': True,
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': True,
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


# ─────────────────────────────────────────────────────────────────────────────
# Custom autograd: L2 distance + similarity with PRP backward
# ─────────────────────────────────────────────────────────────────────────────

class l2_lrp_class(torch.autograd.Function):
    """Custom autograd function that computes L2 distances between conv features
    and prototypes, converts to similarities, and provides a PRP-compatible
    backward pass that distributes relevance based on inverse squared distance."""

    @staticmethod
    def forward(ctx, conv_features, prototype_vectors, ones, proto_activation, epsilon, proto_shape):
        ctx.save_for_backward(conv_features, prototype_vectors)

        x2 = conv_features ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=ones)

        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=conv_features, weight=prototype_vectors)
        intermediate_result = -2 * xp + p2_reshape
        distances = F.relu(x2_patch_sum + intermediate_result)

        if proto_activation == 'log':
            similarities = torch.log((distances + 1) / (distances + epsilon))
        elif proto_activation == 'exp_norm':
            dist_max = proto_shape[1] * proto_shape[2] * proto_shape[3]
            similarities = 1.0 / ((distances / dist_max) + epsilon)
        elif proto_activation == 'linear':
            similarities = -distances
        else:
            raise ValueError(f"Unknown proto_activation: {proto_activation}")

        return similarities

    @staticmethod
    def backward(ctx, grad_output):
        """PRP backward: distribute relevance to each feature channel
        proportional to inverse squared distance to the prototype."""
        conv, prototypes = ctx.saved_tensors
        i = conv.shape[2]
        j = conv.shape[3]
        c = conv.shape[1]
        p = prototypes.shape[0]

        conv_expanded = conv.repeat(p, 1, 1, 1)
        prototype_expanded = prototypes.repeat(1, 1, i, j)
        conv_expanded = conv_expanded.squeeze(0) if conv.shape[0] == 1 else conv_expanded.squeeze()

        l2 = conv_expanded - prototype_expanded
        d = 1.0 / (l2 ** 2 + 1e-12)

        denom = torch.sum(d, dim=1, keepdim=True) + 1e-12
        denom = denom.repeat(1, c, 1, 1) + 1e-12
        R = torch.div(d, denom)

        grad_output_expanded = grad_output.repeat(c, 1, 1, 1)
        grad_output_expanded = grad_output_expanded.permute(1, 0, 2, 3)

        R = R * grad_output_expanded
        R = torch.sum(R, dim=0)
        R = torch.unsqueeze(R, dim=0)

        return R, None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Custom autograd: weighted mean prediction with LRP backward
# ─────────────────────────────────────────────────────────────────────────────

class weighted_mean_lrp(torch.autograd.Function):
    """Custom autograd function for INSightR-Net's weighted mean prediction.

    Forward computes:
        numerator = Σ(activation_i × weight_i²)
        denominator = Σ(activation_i × weight_i² / class_value_i)
        prediction = numerator / denominator

    Backward distributes relevance to each prototype proportional to its
    contribution to the numerator: activation_i × weight_i².
    """

    @staticmethod
    def forward(ctx, prototype_activations, weight_squared, proto_classes, bias):
        numerator = torch.sum(prototype_activations * weight_squared, dim=1, keepdim=True)
        if bias is not None:
            numerator = numerator + bias

        ll_noclass = weight_squared / proto_classes
        denominator = torch.sum(prototype_activations * ll_noclass, dim=1, keepdim=True)

        prediction = numerator / (denominator + 1e-12)

        ctx.save_for_backward(prototype_activations, weight_squared)
        return prediction

    @staticmethod
    def backward(ctx, grad_output):
        activations, weight_squared = ctx.saved_tensors

        contributions = activations * weight_squared
        total = torch.sum(contributions, dim=1, keepdim=True) + 1e-12

        R = (contributions / total) * grad_output
        return R, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Model canonization: convert trained INSightR-Net → LRP-enabled model
# ─────────────────────────────────────────────────────────────────────────────

def PRPCanonizedModel(ppnet, base_arch):
    """Transform a trained INSightR-Net (PPNet) into an LRP-enabled model.

    This function:
    1. Creates a canonized ResNet backbone (fuses BatchNorm, wraps layers with LRP)
    2. Wraps the add-on layers with LRP
    3. Wraps the global max pooling with LRP
    4. Stores metadata needed for PRP heatmap generation

    Args:
        ppnet: Trained INSightR-Net PPNet model instance.
        base_arch: Base architecture string, e.g. 'resnet18'.

    Returns:
        The same ppnet object with all layers replaced by LRP-enabled versions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lrp_params = LRP_PARAMS
    lrp_layer2method = LRP_LAYER2METHOD

    # --- Step 1: Canonize the backbone ---
    canonized_backbone = base_architecture_to_features[base_arch]()
    canonized_backbone = canonized_backbone.to(device)
    canonized_backbone.copyfromresnet(ppnet.features, lrp_params=lrp_params, lrp_layer2method=lrp_layer2method)
    canonized_backbone = canonized_backbone.to(device)
    ppnet.features = canonized_backbone

    # --- Step 2: Canonize the add-on layers ---
    src_modules = list(ppnet.add_on_layers.named_modules())
    first_conv = [m for _, m in src_modules if isinstance(m, nn.Conv2d)][0]
    in_channels = first_conv.in_channels
    out_channels = first_conv.out_channels
    stride = first_conv.stride[0] if isinstance(first_conv.stride, tuple) else first_conv.stride

    canonized_addon = addon_canonized(in_channels, out_channels, stride=stride)
    for src_module_name, src_module in ppnet.add_on_layers.named_modules():
        if isinstance(src_module, nn.Conv2d):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
            setbyname(canonized_addon.addon, src_module_name, wrapped)
        if isinstance(src_module, nn.ReLU):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
            setbyname(canonized_addon.addon, src_module_name, wrapped)
        if isinstance(src_module, nn.Sigmoid):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
            setbyname(canonized_addon.addon, src_module_name, wrapped)

    canonized_addon = canonized_addon.to(device)
    ppnet.add_on_layers = canonized_addon.addon

    # --- Step 3: Create and wrap the global max pooling layer ---
    spatial_size = ppnet.output_size_conv
    ppnet.max_layer = torch.nn.MaxPool2d((spatial_size, spatial_size), return_indices=False)
    ppnet.max_layer = get_lrpwrapperformodule(
        copy.deepcopy(ppnet.max_layer), lrp_params, lrp_layer2method)

    return ppnet


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def relevance_to_heatmap(relevance_tensor, percentile=100):
    """Convert a raw relevance tensor [1, C, H, W] into a 2D heatmap [H, W].

    Returns values normalized to [-1, 1] range.
    """
    hm = relevance_tensor.squeeze().sum(dim=0).detach().cpu().numpy()
    clim = np.percentile(np.abs(hm), percentile)
    if clim > 0:
        hm = hm / clim
    return hm


def invert_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Undo ImageNet normalization on a [C, H, W] or [1, C, H, W] tensor."""
    s = torch.tensor(np.asarray(std, dtype=np.float32)).unsqueeze(1).unsqueeze(2)
    m = torch.tensor(np.asarray(mean, dtype=np.float32)).unsqueeze(1).unsqueeze(2)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor * s + m


# ─────────────────────────────────────────────────────────────────────────────
# PRP heatmap generation functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_prp_per_prototype(inputs, pno, model, device):
    """Generate a PRP heatmap for a single prototype.

    Shows which input pixels caused prototype `pno` to activate.

    Args:
        inputs: Input image tensor [1, 3, H, W].
        pno: Prototype index (0-based).
        model: Canonized INSightR-Net model (output of PRPCanonizedModel).
        device: torch device.

    Returns:
        2D numpy array heatmap normalized to [-1, 1].
    """
    model.train(False)
    inputs = inputs.to(device).clone()
    inputs.requires_grad = True

    with torch.enable_grad():
        conv_features = model.conv_features(inputs)

        similarities = l2_lrp_class.apply(
            conv_features,
            model.prototype_vectors,
            model.ones,
            model.proto_activation,
            model.epsilon,
            model.proto_shape,
        )

        pooled = model.max_layer(similarities)
        pooled = pooled.view(-1, model.num_prototypes)

    pooled[:, pno].backward()

    relevance = inputs.grad.data
    heatmap = relevance_to_heatmap(relevance.to('cpu'))
    return heatmap


def generate_prp_full_prediction(inputs, model, device):
    """Generate a PRP heatmap for the full regression prediction.

    Propagates relevance through the entire weighted-mean computation,
    showing which input pixels contributed to the final prediction value.

    Args:
        inputs: Input image tensor [1, 3, H, W].
        model: Canonized INSightR-Net model (output of PRPCanonizedModel).
        device: torch device.

    Returns:
        Tuple of (heatmap, prediction_value):
          - heatmap: 2D numpy array normalized to [-1, 1].
          - prediction_value: scalar float of the model's prediction.
    """
    model.train(False)
    inputs = inputs.to(device).clone()
    inputs.requires_grad = True

    with torch.enable_grad():
        conv_features = model.conv_features(inputs)

        similarities = l2_lrp_class.apply(
            conv_features,
            model.prototype_vectors,
            model.ones,
            model.proto_activation,
            model.epsilon,
            model.proto_shape,
        )

        pooled = model.max_layer(similarities)
        pooled = pooled.view(-1, model.num_prototypes)

        prototype_activations = pooled

        weight_squared = model.last_layer.weight.data.square().detach()
        proto_classes = model.proto_classes.unsqueeze(0).detach()
        bias = model.last_layer.bias.data.detach() if model.last_layer.bias is not None else None

        prediction = weighted_mean_lrp.apply(
            prototype_activations, weight_squared, proto_classes, bias)

    prediction.backward()

    relevance = inputs.grad.data
    heatmap = relevance_to_heatmap(relevance.to('cpu'))
    pred_value = prediction.item()
    return heatmap, pred_value


def generate_prp_topk_prototypes(inputs, model, device, k=5):
    """Generate PRP heatmaps for the top-k most influential prototypes.

    Prototypes are ranked by their contribution to the prediction:
    contribution_i = activation_i × weight_i² (from the weighted mean numerator).

    Args:
        inputs: Input image tensor [1, 3, H, W].
        model: Canonized INSightR-Net model (output of PRPCanonizedModel).
        device: torch device.
        k: Number of top prototypes.

    Returns:
        Dict with keys:
          - 'prototype_indices': list of top-k prototype indices.
          - 'contributions': list of contribution values.
          - 'heatmaps': list of 2D numpy heatmaps.
          - 'prediction': the model's prediction value.
    """
    model.train(False)

    with torch.no_grad():
        conv_features = model.conv_features(inputs.to(device))

        x2 = conv_features ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=model.ones)
        p2 = model.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3)).view(-1, 1, 1)
        xp = F.conv2d(input=conv_features, weight=model.prototype_vectors)
        distances = F.relu(x2_patch_sum - 2 * xp + p2)

        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size(2), distances.size(3)))
        min_distances = min_distances.view(-1, model.num_prototypes)

        if model.proto_activation == 'log':
            activations = torch.log((min_distances + 1) / (min_distances + model.epsilon))
        elif model.proto_activation == 'exp_norm':
            dist_max = model.proto_shape[1] * model.proto_shape[2] * model.proto_shape[3]
            activations = 1.0 / ((min_distances / dist_max) + model.epsilon)
        elif model.proto_activation == 'linear':
            activations = -min_distances

        weight_squared = model.last_layer.weight.data.square()
        contributions = (activations * weight_squared).squeeze()

        class_idx = model.proto_classes.unsqueeze(0)
        ll_noclass = weight_squared / class_idx
        sum_of_weights = torch.sum(activations * ll_noclass, dim=1)
        numerator = torch.sum(activations * weight_squared, dim=1, keepdim=True)
        if model.last_layer.bias is not None:
            numerator = numerator + model.last_layer.bias.data
        prediction = (numerator / (sum_of_weights.unsqueeze(1) + 1e-12)).item()

        _, topk_indices = torch.topk(contributions.abs(), k=min(k, model.num_prototypes))
        topk_indices = topk_indices.cpu().tolist()
        topk_contributions = contributions[topk_indices].cpu().tolist()

    heatmaps = []
    for pno in topk_indices:
        hm = generate_prp_per_prototype(inputs, pno, model, device)
        heatmaps.append(hm)

    return {
        'prototype_indices': topk_indices,
        'contributions': topk_contributions,
        'heatmaps': heatmaps,
        'prediction': prediction,
    }