"""
INSightR-Net forward pass with selected prototypes *masked out* (zero similarity).

This mirrors the effect of removing prototypes from the weighted
prediction without retraining: basically ablated prototypes contribute zero activation
to both the numerator and denominator of the INSightR-Net prediction formula.
"""

from __future__ import annotations

from typing import Collection

import torch
import torch.nn.functional as F

from insight_training.model import PPNet


def forward_logits_with_prototypes_masked(
    ppnet: PPNet,
    x: torch.Tensor,
    masked_indices: Collection[int],
) -> torch.Tensor:
    """
    Run the same prediction path as ``PPNet.forward``, but set
    ``prototype_activations[:, j] = 0`` for every ``j`` in ``masked_indices``
    before applying the last layer and normalization.

    Args:
        ppnet: Loaded INSightR-Net (eval mode recommended).
        x: Batch of images, shape (N, 3, H, W), same preprocessing as training.
        masked_indices: Prototype indices to exclude from the prediction.
            Empty collection = identical to standard forward (up to float noise).

    Returns:
        Continuous predictions, shape (N,) — same as ``forward(...)[0].squeeze(-1)``.
    """
    if masked_indices:
        bad = [j for j in masked_indices if j < 0 or j >= ppnet.num_prototypes]
        if bad:
            raise ValueError(
                f"masked_indices out of range [0, {ppnet.num_prototypes - 1}]: {bad}"
            )

    distances, _ = ppnet.prototype_distances(x)

    min_distances = -F.max_pool2d(
        -distances,
        kernel_size=(distances.size(2), distances.size(3)),
    )
    min_distances = min_distances.view(-1, ppnet.num_prototypes)
    prototype_activations = ppnet.distance_2_similarity(min_distances)

    if masked_indices:
        pa = prototype_activations.clone()
        for j in masked_indices:
            pa[:, j] = 0.0
    else:
        pa = prototype_activations

    class_id_x = torch.unsqueeze(ppnet.proto_classes, dim=0).to(x.device)
    ll_noclass = ppnet.last_layer.weight.square() / class_id_x
    sum_of_weights = torch.sum(pa * ll_noclass, dim=1, keepdim=True)

    logits = ppnet.last_layer(pa)
    denom = torch.clamp(sum_of_weights, min=1e-12)
    logits = logits / denom
    return logits.squeeze(-1)


def assert_masked_forward_matches_standard(
    ppnet: PPNet,
    x: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """
    Sanity check: empty mask must reproduce ``ppnet.forward``.

    Raises:
        AssertionError if outputs differ beyond tolerances.
    """
    ppnet.eval()
    with torch.no_grad():
        ref, _, _ = ppnet(x, return_convs=False)
        ref = ref.squeeze(-1)
        got = forward_logits_with_prototypes_masked(ppnet, x, ())
    if not torch.allclose(ref, got, atol=atol, rtol=rtol):
        max_diff = (ref - got).abs().max().item()
        raise AssertionError(
            f"Masked forward (empty mask) != standard forward. max |diff| = {max_diff}"
        )

