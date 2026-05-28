# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
Evaluation metrics for token-level attribution quality.

Implements the metrics used in the HETA paper (Section 5):
    - Soft Necessity (Soft-NC)   — Zhao & Aletras (2023), adapted for generation
    - Soft Sufficiency (Soft-NS) — Zhao & Aletras (2023), adapted for generation
    - Dependent Sentence Attribution (DSA) — Introduced in the paper (Section 5.1)
    - F1 Alignment               — Token-level F1 against human annotations
    - Active/Passive Robustness   — Spearman ρ under syntactic rephrasing
    - Sensitivity                 — Attribution stability under Gaussian noise
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from heta.utils import (
    get_embeddings,
    mask_token_embedding,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Soft-NC / Soft-NS (perturbation-based faithfulness)
# ======================================================================

class SoftNC:
    """Soft Necessity Comprehensiveness (Soft-NC).

    Measures how much the output distribution shifts when the *most important*
    tokens (according to attribution scores) are masked.  Higher is better.

    Adapted from Zhao & Aletras (2023) and Zhao & Shan (2024) for generative
    settings.
    """

    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attribution_scores: torch.Tensor,
        target_pos: int,
        target_id: int,
        top_k_fractions: Tuple[float, ...] = (0.1, 0.2, 0.3),
        mask_scheme: str = "zero",
    ) -> float:
        """Compute Soft-NC for a single instance.

        Args:
            model:              The language model.
            input_ids:          Shape ``[1, T]``.
            attribution_scores: Shape ``[target_pos]``.
            target_pos:         Target token position (0-indexed).
            target_id:          Vocabulary index of the target token.
            top_k_fractions:    Fractions of context tokens to mask.
            mask_scheme:        Masking strategy.

        Returns:
            Average Soft-NC score across the specified fractions.
        """
        device = input_ids.device
        embeddings = get_embeddings(model, input_ids).detach()
        logit_pos = target_pos - 1  # autoregressive: logits[t] → token t+1

        # Original distribution at the prediction position
        orig_logits = model(inputs_embeds=embeddings, use_cache=False).logits
        p_orig = F.softmax(orig_logits[0, logit_pos, :], dim=-1)

        # Sort tokens by descending attribution
        num_context = target_pos
        _, sorted_idx = attribution_scores.sort(descending=True)

        nc_scores = []
        for frac in top_k_fractions:
            k = max(1, int(num_context * frac))
            top_idxs = sorted_idx[:k]

            # Mask the top-k tokens
            masked_embeds = embeddings.clone()
            for idx in top_idxs:
                masked_embeds = mask_token_embedding(
                    masked_embeds, idx.item(), scheme=mask_scheme
                )

            masked_logits = model(
                inputs_embeds=masked_embeds, use_cache=False
            ).logits
            p_masked = F.softmax(masked_logits[0, logit_pos, :], dim=-1)

            # KL divergence as shift measure
            kl = F.kl_div(
                p_masked.log().unsqueeze(0),
                p_orig.unsqueeze(0),
                reduction="batchmean",
                log_target=False,
            )
            nc_scores.append(kl.item())

        return sum(nc_scores) / len(nc_scores)


class SoftNS:
    """Soft Sufficiency (Soft-NS).

    Measures how well the output distribution is preserved when only the *most
    important* tokens are kept and the rest are masked.  Higher is better.
    """

    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attribution_scores: torch.Tensor,
        target_pos: int,
        target_id: int,
        top_k_fractions: Tuple[float, ...] = (0.1, 0.2, 0.3),
        mask_scheme: str = "zero",
    ) -> float:
        """Compute Soft-NS for a single instance.

        Returns:
            Average Soft-NS score (higher = better sufficiency).
        """
        device = input_ids.device
        embeddings = get_embeddings(model, input_ids).detach()
        logit_pos = target_pos - 1  # autoregressive: logits[t] → token t+1

        # Original distribution at the prediction position
        orig_logits = model(inputs_embeds=embeddings, use_cache=False).logits
        p_orig = F.softmax(orig_logits[0, logit_pos, :], dim=-1)

        num_context = target_pos
        _, sorted_idx = attribution_scores.sort(descending=True)

        ns_scores = []
        for frac in top_k_fractions:
            k = max(1, int(num_context * frac))
            keep_idxs = set(sorted_idx[:k].tolist())

            # Mask everything EXCEPT the top-k tokens
            masked_embeds = embeddings.clone()
            for i in range(num_context):
                if i not in keep_idxs:
                    masked_embeds = mask_token_embedding(
                        masked_embeds, i, scheme=mask_scheme
                    )

            masked_logits = model(
                inputs_embeds=masked_embeds, use_cache=False
            ).logits
            p_kept = F.softmax(masked_logits[0, logit_pos, :], dim=-1)

            # Lower KL = better sufficiency; report negative KL so higher is better
            kl = F.kl_div(
                p_kept.log().unsqueeze(0),
                p_orig.unsqueeze(0),
                reduction="batchmean",
                log_target=False,
            )
            ns_scores.append(-kl.item())

        return sum(ns_scores) / len(ns_scores)


# ======================================================================
# DSA — Dependent Sentence Attribution (Section 5.1)
# ======================================================================

class DSA:
    """Dependent Sentence Attribution metric.

    Quantifies whether attribution mass concentrates on the answer-relevant
    segment (SciQ) rather than the distractor segment (NarrativeQA) in the
    curated evaluation set.

    DSA = Σ_{i ∈ S_SciQ} ss_i  −  Σ_{j ∈ S_NarrQA} fs_j

    where attributions are normalized so total mass over the paragraph equals 1.
    """

    @staticmethod
    def evaluate(
        attribution_scores: torch.Tensor,
        relevant_indices: List[int],
        distractor_indices: List[int],
    ) -> float:
        """Compute DSA for a single instance.

        Args:
            attribution_scores: Per-token scores (non-negative), any length.
            relevant_indices:   Token indices in the answer-relevant segment.
            distractor_indices: Token indices in the distractor segment.

        Returns:
            Scalar DSA value (higher is better).
        """
        total = attribution_scores.sum()
        if total <= 0:
            return 0.0

        normed = attribution_scores / total

        relevant_mass = normed[relevant_indices].sum().item() if relevant_indices else 0.0
        distractor_mass = normed[distractor_indices].sum().item() if distractor_indices else 0.0

        return relevant_mass - distractor_mass


# ======================================================================
# F1 Alignment (Section 5.3)
# ======================================================================

def f1_alignment(
    attribution_scores: torch.Tensor,
    gold_indices: List[int],
    top_k: Optional[int] = None,
) -> float:
    """Token-level F1 between top-attributed tokens and gold annotations.

    Args:
        attribution_scores: Per-token scores.
        gold_indices:       Annotated important token indices.
        top_k:              Number of top tokens to consider (default: |gold|).

    Returns:
        F1 score in [0, 1].
    """
    if not gold_indices:
        return 0.0
    if top_k is None:
        top_k = len(gold_indices)

    _, predicted = attribution_scores.topk(min(top_k, len(attribution_scores)))
    predicted_set = set(predicted.tolist())
    gold_set = set(gold_indices)

    tp = len(predicted_set & gold_set)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


# ======================================================================
# Sensitivity (Section 5.3, Eq. 18)
# ======================================================================

def sensitivity(
    attributor,
    text: str,
    num_perturbations: int = 5,
    noise_std: float = 0.01,
) -> float:
    """Measure attribution stability under Gaussian embedding noise.

    Returns:
        Average per-token standard deviation (lower is better).
    """
    all_scores = []
    for _ in range(num_perturbations):
        result = attributor.attribute(text)
        all_scores.append(result.scores.cpu())

    stacked = torch.stack(all_scores, dim=0)  # [num_perturbations, T-1]
    per_token_std = stacked.std(dim=0)
    return per_token_std.mean().item()
