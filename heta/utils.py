# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""Shared utilities for the HETA attribution framework."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HETAConfig:
    """Configuration for the HETA attribution pipeline.

    Attributes:
        beta:           Weight for Hessian-based sensitivity (Eq. 5).
        gamma:          Weight for KL information contribution (Eq. 5).
        num_hvp_samples: Number of Rademacher vectors for Hutchinson HVP (m).
        mask_scheme:    Token masking strategy — ``"zero"`` | ``"mean"`` | ``"unk"``.
        window_size:    Sliding-window length for long-context attribution (0 = off).
        window_overlap: Fractional overlap between consecutive windows.
        low_rank:       Rank for low-rank Hessian approximation (0 = full).
        layer_subset:   If > 0, compute curvature only on the last *n* layers.
        use_fisher:     Replace exact Hessian with Gauss–Newton / Fisher surrogate.
        device:         Target device (``"cuda"`` | ``"cpu"`` | ``"auto"``).
        dtype:          Computation dtype (``torch.float32`` recommended).
    """

    beta: float = 0.5
    gamma: float = 0.5
    num_hvp_samples: int = 10
    mask_scheme: str = "zero"
    window_size: int = 0
    window_overlap: float = 0.5
    low_rank: int = 0
    layer_subset: int = 0
    use_fisher: bool = False
    device: str = "auto"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        assert 0.0 <= self.beta, "beta must be non-negative."
        assert 0.0 <= self.gamma, "gamma must be non-negative."
        assert self.num_hvp_samples >= 1, "Need at least 1 HVP sample."
        assert self.mask_scheme in {"zero", "mean", "unk"}, (
            f"Unknown mask_scheme={self.mask_scheme!r}."
        )
        if self.window_size > 0:
            assert 0.0 <= self.window_overlap < 1.0, (
                "window_overlap must be in [0, 1)."
            )


# ---------------------------------------------------------------------------
# Attribution result container
# ---------------------------------------------------------------------------

@dataclass
class AttributionResult:
    """Container for HETA attribution outputs.

    Attributes:
        scores:             Final per-token attribution (Eq. 5), shape ``[T-1]``.
        semantic_flow:      Semantic transition vector M_T, shape ``[T-1]``.
        hessian_sensitivity: Hessian sensitivity S_i^(T), shape ``[T-1]``.
        kl_divergence:      KL information impact I(x_i -> x_T), shape ``[T-1]``.
        tokens:             Decoded token strings for display.
        target_token:       The target token string.
        target_position:    Index of the target position T in the sequence.
    """

    scores: torch.Tensor
    semantic_flow: torch.Tensor
    hessian_sensitivity: torch.Tensor
    kl_divergence: torch.Tensor
    tokens: List[str] = field(default_factory=list)
    target_token: str = ""
    target_position: int = -1

    def topk(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-*k* attributed tokens and their scores."""
        vals, idxs = self.scores.topk(min(k, len(self.scores)))
        return [(self.tokens[i], v.item()) for i, v in zip(idxs.tolist(), vals)]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    """Resolve ``'auto'`` to the best available device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_embeddings(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract input embeddings from the model's embedding layer.

    Args:
        model:     A HuggingFace ``PreTrainedModel``.
        input_ids: Token IDs, shape ``[1, T]``.

    Returns:
        Embeddings tensor of shape ``[1, T, d]`` with ``requires_grad=True``.
    """
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).detach().clone()
    embeds.requires_grad_(True)
    return embeds


def forward_with_embeddings(
    model: PreTrainedModel,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple:
    """Run a forward pass using raw embeddings instead of ``input_ids``.

    Returns:
        Tuple of (logits, attentions, hidden_states).
          - logits:        ``[1, T, V]``
          - attentions:    tuple of ``[1, H, T, T]`` per layer
          - hidden_states: tuple of ``[1, T, d]`` per layer (layer 0 = embeddings)
    """
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.logits, outputs.attentions, outputs.hidden_states


def get_target_log_prob(
    logits: torch.Tensor,
    target_id: int,
    target_pos: int,
) -> torch.Tensor:
    """Compute log P(x_T | x_{<T}) for a specific target token.

    In autoregressive LMs, ``logits[0, t, :]`` predicts the token at position
    ``t + 1``.  Therefore, to obtain ``P(x_T | x_{<T})`` for the target at
    position ``target_pos``, we read logits at ``target_pos - 1``.

    Args:
        logits:     Model logits, shape ``[1, T, V]``.
        target_id:  Vocabulary index of the target token.
        target_pos: 0-indexed position of the target token in the sequence.

    Returns:
        Scalar log-probability tensor with gradient graph attached.
    """
    logit_pos = target_pos - 1  # logits at t predict token at t+1
    log_probs = F.log_softmax(logits[0, logit_pos, :], dim=-1)
    return log_probs[target_id]


def mask_token_embedding(
    embeddings: torch.Tensor,
    token_idx: int,
    scheme: str = "zero",
) -> torch.Tensor:
    """Return a copy of *embeddings* with one token masked.

    Args:
        embeddings: Shape ``[1, T, d]``.
        token_idx:  Index of the token to mask.
        scheme:     ``"zero"`` | ``"mean"`` | ``"unk"``.

    Returns:
        New tensor of same shape with the specified token replaced.
    """
    masked = embeddings.clone()
    if scheme == "zero":
        masked[0, token_idx, :] = 0.0
    elif scheme == "mean":
        masked[0, token_idx, :] = embeddings[0].mean(dim=0)
    elif scheme == "unk":
        # Replace with a small random vector (sentinel)
        masked[0, token_idx, :] = torch.randn_like(masked[0, token_idx, :]) * 0.01
    return masked


def build_windows(
    seq_len: int,
    window_size: int,
    overlap: float,
) -> List[Tuple[int, int]]:
    """Generate overlapping window boundaries for long-context attribution.

    Args:
        seq_len:     Total context length (T - 1, excluding the target).
        window_size: Number of tokens per window.
        overlap:     Fractional overlap in ``[0, 1)``.

    Returns:
        List of ``(start, end)`` pairs covering the full context.
    """
    if window_size <= 0 or window_size >= seq_len:
        return [(0, seq_len)]

    stride = max(1, int(window_size * (1.0 - overlap)))
    windows = []
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        windows.append((start, end))
        if end == seq_len:
            break
        start += stride
    return windows
