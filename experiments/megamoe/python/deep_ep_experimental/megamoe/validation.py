# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Routing and collective preflight checks for MegaMoE."""

import math
from deep_ep._store import StoreGroup


def check_routing(indices, weights, num_experts, num_topk):
    """Validate CPU lists obtained from the actual input tensors."""
    if len(indices) != len(weights):
        raise ValueError("Routing IDs and weights have different BS")
    for token, (ids, probs) in enumerate(zip(indices, weights)):
        if len(ids) != num_topk or len(probs) != num_topk:
            raise ValueError(f"token={token}: routing width must equal top-k={num_topk}")
        if any(type(expert) is not int or not 0 <= expert < num_experts for expert in ids):
            raise ValueError(f"token={token}: expert IDs must be in [0,{num_experts}); -1 masking is unsupported")
        if len(set(ids)) != num_topk:
            raise ValueError(f"token={token}: duplicate expert IDs are unsupported")
        if any(not math.isfinite(value) for value in probs):
            raise ValueError(f"token={token}: route weights must be finite")


def check_window_routes(indices, weights, num_experts):
    """Check the entire logical batch before any window starts exchanging payloads."""
    import torch

    ordered = indices.sort(dim=1).values
    invalid_id = (ordered[:, 0] < 0).any() | (ordered[:, -1] >= num_experts).any()
    failures = [(invalid_id, f"expert IDs must be in [0,{num_experts})")]
    if ordered.shape[1] > 1:
        duplicate = (ordered[:, 1:] == ordered[:, :-1]).any()
        failures.append((duplicate, "duplicate expert IDs are unsupported"))
    failures.append((~torch.isfinite(weights).all(), "route weights must be finite"))
    flags = torch.stack([flag for flag, _ in failures]).cpu().tolist()
    for failed, (_, reason) in zip(flags, failures):
        if failed:
            raise ValueError(reason)


def gather_control(group, stage, value):
    if isinstance(group, StoreGroup):
        return group.all_gather(value, stage)
    import torch.distributed as dist

    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, value, group=group)
    return peers


def collective_check(group, stage, error, signature=None):
    peers = gather_control(group, f"check:{stage}", (error, signature))
    failures = [f"rank={rank}: {value[0]}" for rank, value in enumerate(peers) if value[0]]
    if failures:
        raise ValueError(f"MegaMoE {stage} failed: " + "; ".join(failures))
    if any(value[1] != peers[0][1] for value in peers):
        raise ValueError(
            f"MegaMoE {stage}: ranks disagree: "
            + "; ".join(f"rank={rank}: {value[1]}" for rank, value in enumerate(peers))
        )
