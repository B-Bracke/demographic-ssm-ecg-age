"""Data augmentation utilities for ECG training."""

import numpy as np
import torch


def get_channel_mask(traces, drop_prob=0.33, max_drop=11):
    """Generate a binary mask for random lead (channel) dropout.

    Args:
        traces: Input tensor of shape [B, leads, seq_len] (used for shape/device).
        drop_prob: Probability of applying dropout to the batch.
        max_drop: Maximum number of leads to drop per sample.

    Returns:
        mask: Tensor of shape [B, leads] (1.0=keep, 0.0=drop), or None if skipped.
    """
    if np.random.rand() > drop_prob:
        return None

    bs, n_leads, _ = traces.shape
    device = traces.device
    mask = torch.ones((bs, n_leads), device=device)

    for i in range(bs):
        n_drop = np.random.randint(1, max_drop + 1)
        drop_indices = np.random.choice(n_leads, n_drop, replace=False)
        mask[i, drop_indices] = 0.0

    return mask


def apply_time_masking(traces, prob=0.5, max_mask_ratio=0.2, num_masks=2):
    """Zero out contiguous time segments of the ECG signal (SpecAugment-style).

    Masks are applied consistently across all leads for each sample.

    Args:
        traces: Tensor of shape [B, leads, seq_len].
        prob: Per-sample probability of applying masking.
        max_mask_ratio: Maximum fraction of the signal length to mask per segment.
        num_masks: Number of independent mask segments per sample.

    Returns:
        traces: Augmented tensor (modified in-place).
    """
    if prob <= 0:
        return traces

    bs, n_leads, seq_len = traces.shape

    for i in range(bs):
        if np.random.rand() > prob:
            continue
        for _ in range(num_masks):
            mask_len = np.random.randint(0, int(seq_len * max_mask_ratio))
            start = np.random.randint(0, seq_len - mask_len)
            end = start + mask_len
            traces[i, :, start:end] = 0.0

    return traces
