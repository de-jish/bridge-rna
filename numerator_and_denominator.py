# coding=utf-8
"""Numerator/denominator prefix-sum ops used by SLiMPerformer.

This local implementation covers inference paths used by the Dash retrieval app.
It provides the API expected by slim_performer_model.py.
"""

from __future__ import annotations

import torch


def _ensure_batch_sums(sum_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if sum_tensor.shape[0] == batch_size:
        return sum_tensor
    if sum_tensor.shape[0] == 1:
        return sum_tensor.expand(batch_size, *sum_tensor.shape[1:]).clone()
    raise ValueError(
        f"Invalid prefix-sum batch dimension: got {sum_tensor.shape[0]}, expected 1 or {batch_size}"
    )


def num_iter(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_sums: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterative causal numerator accumulation.

    queries: [B, T, H, F]
    keys: [B, T, H, F]
    values: [B, T, H, D]
    num_sums: [1|B, H, F, D]
    returns num: [T, B, H, D], new_num_sums: [B, H, F, D]
    """
    bsz, tlen, _, _ = queries.shape
    acc = _ensure_batch_sums(num_sums, bsz)
    outs = []

    for t in range(tlen):
        k_t = keys[:, t]     # [B, H, F]
        v_t = values[:, t]   # [B, H, D]
        acc = acc + torch.einsum("bhf,bhd->bhfd", k_t, v_t)
        q_t = queries[:, t]  # [B, H, F]
        out_t = torch.einsum("bhf,bhfd->bhd", q_t, acc)
        outs.append(out_t)

    return torch.stack(outs, dim=0), acc


def den_iter(
    queries: torch.Tensor,
    keys: torch.Tensor,
    den_sums: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterative causal denominator accumulation.

    queries: [B, T, H, F]
    keys: [B, T, H, F]
    den_sums: [1|B, H, F]
    returns den: [T, B, H], new_den_sums: [B, H, F]
    """
    bsz, tlen, _, _ = queries.shape
    acc = _ensure_batch_sums(den_sums, bsz)
    outs = []

    for t in range(tlen):
        k_t = keys[:, t]    # [B, H, F]
        acc = acc + k_t
        q_t = queries[:, t]  # [B, H, F]
        out_t = torch.einsum("bhf,bhf->bh", q_t, acc)
        outs.append(out_t)

    return torch.stack(outs, dim=0), acc


def num_ps(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_sums: torch.Tensor,
    _parallel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep API compatibility; inference can reuse iterative path.
    return num_iter(queries, keys, values, num_sums)


def den_ps(
    queries: torch.Tensor,
    keys: torch.Tensor,
    den_sums: torch.Tensor,
    _parallel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep API compatibility; inference can reuse iterative path.
    return den_iter(queries, keys, den_sums)


def num_reverse_sums_iter(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_sums: torch.Tensor,
) -> torch.Tensor:
    # Minimal compatibility for training-oriented call sites.
    bsz = queries.shape[0]
    acc = _ensure_batch_sums(num_sums, bsz)
    for t in range(queries.shape[1] - 1, -1, -1):
        acc = acc + torch.einsum("bhf,bhd->bhfd", keys[:, t], values[:, t])
    return acc


def den_reverse_sums_iter(
    queries: torch.Tensor,
    keys: torch.Tensor,
    den_sums: torch.Tensor,
) -> torch.Tensor:
    # Minimal compatibility for training-oriented call sites.
    bsz = queries.shape[0]
    acc = _ensure_batch_sums(den_sums, bsz)
    for t in range(queries.shape[1] - 1, -1, -1):
        acc = acc + keys[:, t]
    return acc


def num_reverse_sums_ps(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_sums: torch.Tensor,
) -> torch.Tensor:
    return num_reverse_sums_iter(queries, keys, values, num_sums)


def den_reverse_sums_ps(
    queries: torch.Tensor,
    keys: torch.Tensor,
    den_sums: torch.Tensor,
) -> torch.Tensor:
    return den_reverse_sums_iter(queries, keys, den_sums)
