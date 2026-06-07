"""
Pruned PRP code for the CleverHansRegression PRP-INSightR-Net pipeline.

Simple steps for PLRP-lambda pruning:
  1. Sort positive entries of R ascending; find threshold theta_pos such that
     entries with R_i <= theta_pos sum to <= p_pos * total_positive_mass.
  2. Sort |negative| entries of R ascending; find threshold theta_neg such that
     entries with |R_i| <= theta_neg sum to <= p_neg * total_negative_mass.
  3. Zero out entries with theta_neg < R_i <= theta_pos, then rescale the
     remaining positive (also negative) entries by lambda = total / kept_sum
     so that for each layer positive (and also negative) mass is preserved.

When p_pos == p_neg == 0 the we get back result like normal PRP.

In main PLRP paper, the first layer is left unpruned. We follow that.
"""

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# PLRP-lambda state and helper
# ──────────────────────────────────────────────────────────

_PLRP_STATE = {
    'p_pos': 0.0,
    'p_neg': 0.0,
    'enabled': False,
}


def set_plrp_params(p_pos: float = 0.0, p_neg: float = 0.0) -> None:
    """Configure PLRP-lambda pruning for subsequent backward passes.

    ``p_pos`` is the proportion of positive relevance mass to prune at each
    parametric layer. ``p_neg`` is the proportion of negative relevance mass
    to prune. Both values are within 0 and 1.
    """
    if not (0.0 <= p_pos < 1.0):
        raise ValueError(f"p_pos must be in [0, 1); got {p_pos}")
    if not (0.0 <= p_neg < 1.0):
        raise ValueError(f"p_neg must be in [0, 1); got {p_neg}")
    _PLRP_STATE['p_pos'] = float(p_pos)
    _PLRP_STATE['p_neg'] = float(p_neg)
    _PLRP_STATE['enabled'] = (p_pos > 0.0) or (p_neg > 0.0)


def get_plrp_params() -> tuple:
    """Return the currently configured ``(p_pos, p_neg)`` pruning proportions."""
    return _PLRP_STATE['p_pos'], _PLRP_STATE['p_neg']


def _prune_relevance_lambda(R: torch.Tensor, p_pos: float, p_neg: float,
                            eps: float = 1e-12) -> torch.Tensor:
    """PLRP-lambda: per-sample threshold + rescale of a relevance tensor.
    """
    if (p_pos <= 0.0) and (p_neg <= 0.0):
        return R
    if R.numel() == 0:
        return R

    flat = R.reshape(R.shape[0], -1)
    out = flat.clone()

    for b in range(flat.shape[0]):
        v = flat[b]

        if p_pos > 0.0:
            pos_mask = (v > 0)
            if pos_mask.any():
                pos_vals = v[pos_mask]
                sorted_pos, _ = torch.sort(pos_vals)
                cumsum = torch.cumsum(sorted_pos, dim=0)
                target = p_pos * cumsum[-1]
                below_or_eq = cumsum <= target
                if below_or_eq.any():
                    last_idx = int(below_or_eq.nonzero(as_tuple=False).max().item())
                    theta = sorted_pos[last_idx]
                else:
                    theta = torch.tensor(float('-inf'), device=v.device, dtype=v.dtype)
                keep_pos = pos_mask & (v > theta)
                prune_pos = pos_mask & (v <= theta)
                kept_sum = v[keep_pos].sum()
                pos_total = cumsum[-1]
                if kept_sum > eps:
                    lam_pos = pos_total / kept_sum
                else:
                    lam_pos = torch.tensor(1.0, device=v.device, dtype=v.dtype)
                out[b] = torch.where(prune_pos, torch.zeros_like(v), out[b])
                out[b] = torch.where(keep_pos, v * lam_pos, out[b])

        if p_neg > 0.0:
            neg_mask = (v < 0)
            if neg_mask.any():
                neg_abs = -v[neg_mask]
                sorted_abs, _ = torch.sort(neg_abs)
                cumsum = torch.cumsum(sorted_abs, dim=0)
                target = p_neg * cumsum[-1]
                below_or_eq = cumsum <= target
                if below_or_eq.any():
                    last_idx = int(below_or_eq.nonzero(as_tuple=False).max().item())
                    theta_abs = sorted_abs[last_idx]
                else:
                    theta_abs = torch.tensor(float('-inf'), device=v.device, dtype=v.dtype)
                keep_neg = neg_mask & (v < -theta_abs)
                prune_neg = neg_mask & (v >= -theta_abs)
                kept_abs_sum = (-v[keep_neg]).sum() if keep_neg.any() else torch.tensor(0.0, device=v.device, dtype=v.dtype)
                neg_total = cumsum[-1]
                if kept_abs_sum > eps:
                    lam_neg = neg_total / kept_abs_sum
                else:
                    lam_neg = torch.tensor(1.0, device=v.device, dtype=v.dtype)
                out[b] = torch.where(prune_neg, torch.zeros_like(v), out[b])
                out[b] = torch.where(keep_neg, v * lam_neg, out[b])

    return out.reshape_as(R)


def _maybe_prune(R: torch.Tensor) -> torch.Tensor:
    """Apply PLRP-lambda pruning if currently enabled, else return R unchanged."""
    if not _PLRP_STATE['enabled']:
        return R
    return _prune_relevance_lambda(R, _PLRP_STATE['p_pos'], _PLRP_STATE['p_neg'])


# ──────────────────────────────────────────────────────────
# Wrapper classes for wrapping standard modules with LRP
# ──────────────────────────────────────────────────────────

class zeroparam_wrapper_class(nn.Module):
    def __init__(self, module, autogradfunction):
        super().__init__()
        self.module = module
        self.wrapper = autogradfunction

    def forward(self, x):
        return self.wrapper.apply(x, self.module)


class oneparam_wrapper_class(nn.Module):
    def __init__(self, module, autogradfunction, parameter1):
        super().__init__()
        self.module = module
        self.wrapper = autogradfunction
        self.parameter1 = parameter1

    def forward(self, x):
        return self.wrapper.apply(x, self.module, self.parameter1)


class conv2d_zbeta_wrapper_class(nn.Module):
    def __init__(self, module, lrpignorebias, lowest=None, highest=None):
        super().__init__()
        if lowest is None:
            lowest = torch.tensor(0.0)
        if highest is None:
            highest = torch.tensor(1.0)
        assert isinstance(module, nn.Conv2d)
        self.module = module
        self.wrapper = conv2d_zbeta_wrapper_fct()
        self.lrpignorebias = lrpignorebias
        self.lowest = lowest
        self.highest = highest

    def forward(self, x):
        return self.wrapper.apply(x, self.module, self.lrpignorebias, self.lowest, self.highest)


# ──────────────────────────────────────────────────────────
# Module-to-wrapper lookup
# ──────────────────────────────────────────────────────────

class lrplookupnotfounderror(Exception):
    pass


def get_lrpwrapperformodule(module, lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta=False):

    if isinstance(module, nn.ReLU):
        key = 'nn.ReLU'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

    elif isinstance(module, nn.Sigmoid):
        key = 'nn.Sigmoid'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

    elif isinstance(module, nn.BatchNorm2d):
        key = 'nn.BatchNorm2d'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

    elif isinstance(module, nn.Linear):
        key = 'nn.Linear'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['linear_eps'])

    elif isinstance(module, nn.Conv2d):
        if thisis_inputconv_andiwant_zbeta:
            return conv2d_zbeta_wrapper_class(
                module, lrp_params['conv2d_ignorebias'],
                lowest=lrp_params.get('lowest', None),
                highest=lrp_params.get('highest', None),
            )
        else:
            key = 'nn.Conv2d'
            if key not in lrp_layer2method:
                raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
            autogradfunction = lrp_layer2method[key]()
            return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['conv2d_ignorebias'])

    elif isinstance(module, nn.AdaptiveAvgPool2d):
        key = 'nn.AdaptiveAvgPool2d'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['pooling_eps'])

    elif isinstance(module, nn.AvgPool2d):
        key = 'nn.AvgPool2d'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['pooling_eps'])

    elif isinstance(module, nn.MaxPool2d):
        key = 'nn.MaxPool2d'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return zeroparam_wrapper_class(module, autogradfunction=autogradfunction)

    elif isinstance(module, sum_stacked2):
        key = 'sum_stacked2'
        if key not in lrp_layer2method:
            raise lrplookupnotfounderror(f"No entry in lrp_layer2method for: {key}")
        autogradfunction = lrp_layer2method[key]()
        return oneparam_wrapper_class(module, autogradfunction=autogradfunction, parameter1=lrp_params['eltwise_eps'])

    else:
        raise lrplookupnotfounderror(f"No lookup for module: {module}")


# ──────────────────────────────────────────────────────────
# Canonization: fuse BatchNorm into Conv
# ──────────────────────────────────────────────────────────

def resetbn(bn):
    assert isinstance(bn, nn.BatchNorm2d)
    bnc = copy.deepcopy(bn)
    bnc.reset_parameters()
    return bnc


def bnafterconv_overwrite_intoconv(conv, bn):
    assert isinstance(bn, nn.BatchNorm2d)
    assert isinstance(conv, nn.Conv2d)

    s = (bn.running_var + bn.eps) ** 0.5
    w = bn.weight
    b = bn.bias
    m = bn.running_mean

    conv.weight = torch.nn.Parameter(conv.weight * (w / s).reshape(-1, 1, 1, 1))
    if conv.bias is None:
        conv.bias = torch.nn.Parameter((0 - m) * (w / s) + b)
    else:
        conv.bias = torch.nn.Parameter((conv.bias - m) * (w / s) + b)
    return conv


# ──────────────────────────────────────────────────────────
# ResNet helper modules
# ──────────────────────────────────────────────────────────

class sum_stacked2(nn.Module):
    """Sums two tensors stacked along dim=0 (for residual connections)."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        assert x.shape[0] == 2
        return torch.sum(x, dim=0)


# ──────────────────────────────────────────────────────────
# Positive/negative weight decomposition for LRP rules
# ──────────────────────────────────────────────────────────

class posnegconv(nn.Module):
    """Decomposes convolution into positive and negative parts for z+ rule."""

    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super().__init__()
        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)
        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)

        if ignorebias:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0))
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0))

    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn


class anysign_conv(nn.Module):
    """Convolution decomposed into positive/negative/full modes for z-beta rule."""

    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super().__init__()
        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)
        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)
        self.jusconv = self._clone_module(conv)
        self.jusconv.weight = torch.nn.Parameter(conv.weight.data.clone()).to(conv.weight.device)

        if ignorebias:
            self.posconv.bias = None
            self.negconv.bias = None
            self.jusconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0)).to(conv.weight.device)
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0)).to(conv.weight.device)
                self.jusconv.bias = torch.nn.Parameter(conv.bias.data.clone()).to(conv.weight.device)

    def forward(self, mode, x):
        if mode == 'pos':
            return self.posconv.forward(x)
        elif mode == 'neg':
            return self.negconv.forward(x)
        elif mode == 'justasitis':
            return self.jusconv.forward(x)
        else:
            raise NotImplementedError(f"anysign_conv: unknown mode '{mode}'")


# ──────────────────────────────────────────────────────────
# Base LRP routines
# ──────────────────────────────────────────────────────────

def safe_divide(numerator, divisor, eps0, eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign())


def lrp_backward(_input, layer, relevance_output, eps0, eps):
    """Core LRP backward: forward through layer, then redistribute relevance."""
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(_input)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    Z.backward(S)
    relevance_input = _input.data * _input.grad.data
    return relevance_input


# ──────────────────────────────────────────────────────────
# Autograd functions: custom backward passes implementing LRP rules
# ──────────────────────────────────────────────────────────

def _conv2d_config_to_tensorlist(module):
    propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
    values = []
    for attr in propertynames:
        v = getattr(module, attr)
        if isinstance(v, int):
            v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
        elif isinstance(v, tuple):
            v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
        else:
            raise ValueError(f'Unexpected type for {attr}: {type(v)}')
        values.append(v)
    return propertynames, values


def _tensorlist_to_conv2d_dict(values):
    propertynames = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
    paramsdict = {}
    for i, n in enumerate(propertynames):
        v = values[i]
        alist = v.tolist()
        if len(alist) == 1:
            paramsdict[n] = alist[0]
        else:
            paramsdict[n] = tuple(alist)
    return paramsdict


class conv2d_beta0_wrapper_fct(torch.autograd.Function):
    """LRP z+ rule for Conv2d layers (PLRP-lambda pruned when enabled)."""

    @staticmethod
    def forward(ctx, x, module, lrpignorebias):
        _, values = _conv2d_config_to_tensorlist(module)
        bias = None if module.bias is None else module.bias.data.clone()
        lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool, device=module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor, *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, lrpignorebiastensor, *values = ctx.saved_tensors
        paramsdict = _tensorlist_to_conv2d_dict(values)

        if conv2dbias is None:
            module = nn.Conv2d(**paramsdict, bias=False)
        else:
            module = nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)
        module.weight = torch.nn.Parameter(conv2dweight)

        pnconv = posnegconv(module, ignorebias=lrpignorebiastensor.item())
        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=pnconv, relevance_output=grad_output[0], eps0=1e-12, eps=0)
        R = _maybe_prune(R)
        return R, None, None


class conv2d_zbeta_wrapper_fct(torch.autograd.Function):
    """LRP z-beta rule for the first Conv2d layer.

    The first (input-touching) layer is intentionally NOT pruned, because it also suggested in PLRP
    paper
    """

    @staticmethod
    def forward(ctx, x, module, lrpignorebias, lowest, highest):
        _, values = _conv2d_config_to_tensorlist(module)
        bias = None if module.bias is None else module.bias.data.clone()
        lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool, device=module.weight.device)
        ctx.save_for_backward(x, module.weight.data.clone(), bias, lrpignorebiastensor,
                              lowest.to(module.weight.device), highest.to(module.weight.device), *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, conv2dweight, conv2dbias, lrpignorebiastensor, lowest_, highest_, *values = ctx.saved_tensors
        paramsdict = _tensorlist_to_conv2d_dict(values)

        if conv2dbias is None:
            module = nn.Conv2d(**paramsdict, bias=False)
        else:
            module = nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)
        module.weight = torch.nn.Parameter(conv2dweight)

        any_conv = anysign_conv(module, ignorebias=lrpignorebiastensor.item())
        X = input_.clone().detach().requires_grad_(True)
        L = (lowest_ * torch.ones_like(X)).requires_grad_(True)
        H = (highest_ * torch.ones_like(X)).requires_grad_(True)

        with torch.enable_grad():
            Z = any_conv.forward(mode='justasitis', x=X) - any_conv.forward(mode='pos', x=L) - any_conv.forward(mode='neg', x=H)
            S = safe_divide(grad_output[0].clone().detach(), Z.clone().detach(), eps0=1e-6, eps=1e-6)
            Z.backward(S)
            R = (X * X.grad + L * L.grad + H * H.grad).detach()
        return R, None, None, None, None


class relu_wrapper_fct(torch.autograd.Function):
    """LRP pass-through for ReLU (relevance passes unchanged; not pruned)."""

    @staticmethod
    def forward(ctx, x, module):
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class sigmoid_wrapper_fct(torch.autograd.Function):
    """LRP pass-through for Sigmoid (relevance passes unchanged; not pruned)."""

    @staticmethod
    def forward(ctx, x, module):
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class linearlayer_eps_wrapper_fct(torch.autograd.Function):
    """LRP epsilon rule for Linear layers (PLRP-lambda pruned when enabled)."""

    @staticmethod
    def forward(ctx, x, module, eps):
        propertynames = ['in_features', 'out_features']
        values = []
        for attr in propertynames:
            v = getattr(module, attr)
            if isinstance(v, int):
                v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
            elif isinstance(v, tuple):
                v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
            values.append(v)

        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
        bias = None if module.bias is None else module.bias.data.clone()
        ctx.save_for_backward(x, module.weight.data.clone(), bias, epstensor, *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, epstensor, *values = ctx.saved_tensors
        propertynames = ['in_features', 'out_features']
        paramsdict = {}
        for i, n in enumerate(propertynames):
            v = values[i]
            alist = v.tolist()
            paramsdict[n] = alist[0] if len(alist) == 1 else tuple(alist)

        if bias is None:
            module = nn.Linear(**paramsdict, bias=False)
        else:
            module = nn.Linear(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(weight)

        eps = epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=module, relevance_output=grad_output[0], eps0=eps, eps=eps)
        R = _maybe_prune(R)
        return R, None, None


class adaptiveavgpool2d_wrapper_fct(torch.autograd.Function):
    """LRP for AdaptiveAvgPool2d (not pruned: not a parametric layer in PLRP)."""

    @staticmethod
    def forward(ctx, x, module, eps):
        v = getattr(module, 'output_size')
        if isinstance(v, int):
            v = torch.tensor([v], dtype=torch.int32, device=x.device)
        elif isinstance(v, tuple):
            v = torch.tensor(v, dtype=torch.int32, device=x.device)
        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
        ctx.save_for_backward(x, epstensor, v)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, epstensor, v = ctx.saved_tensors
        alist = v.tolist()
        output_size = alist[0] if len(alist) == 1 else tuple(alist)
        eps = epstensor.item()
        layerclass = torch.nn.AdaptiveAvgPool2d(output_size)
        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=layerclass, relevance_output=grad_output[0], eps0=eps, eps=eps)
        return R, None, None


class maxpool2d_wrapper_fct(torch.autograd.Function):
    """LRP for MaxPool2d (not pruned: not a parametric layer in PLRP)."""

    @staticmethod
    def forward(ctx, x, module):
        propertynames = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
        values = []
        for attr in propertynames:
            v = getattr(module, attr)
            if isinstance(v, bool):
                v = torch.tensor([v], dtype=torch.bool, device=x.device)
            elif isinstance(v, int):
                v = torch.tensor([v], dtype=torch.int32, device=x.device)
            elif isinstance(v, tuple):
                v = torch.tensor(v, dtype=torch.int32, device=x.device)
            else:
                raise ValueError(f'Unexpected type for {attr}: {type(v)}')
            values.append(v)
        ctx.save_for_backward(x, *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, *values = ctx.saved_tensors
        propertynames = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
        paramsdict = {}
        for i, n in enumerate(propertynames):
            v = values[i]
            alist = v.tolist()
            paramsdict[n] = alist[0] if len(alist) == 1 else tuple(alist)

        layerclass = torch.nn.MaxPool2d(**paramsdict)
        X = input_.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            Z = layerclass.forward(X)
        relevance_output_data = grad_output[0].clone().detach().unsqueeze(0)
        Z.backward(relevance_output_data)
        R = X.grad
        return R, None


class eltwisesum_stacked2_eps_wrapper_fct(torch.autograd.Function):
    """LRP for element-wise sum in residual connections (not pruned)."""

    @staticmethod
    def forward(ctx, stackedx, module, eps):
        epstensor = torch.tensor([eps], dtype=torch.float32, device=stackedx.device)
        ctx.save_for_backward(stackedx, epstensor)
        return module.forward(stackedx)

    @staticmethod
    def backward(ctx, grad_output):
        stackedx, epstensor = ctx.saved_tensors
        X = stackedx.clone().detach().requires_grad_(True)
        eps = epstensor.item()
        s2 = sum_stacked2().to(X.device)
        Rtmp = lrp_backward(_input=X, layer=s2, relevance_output=grad_output[0], eps0=eps, eps=eps)
        return Rtmp, None, None
