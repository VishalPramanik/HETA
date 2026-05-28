# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
KL-Based Information Impact — Component (3) of HETA.

Measures how the predictive distribution at the target position changes when
each context token is individually masked.  This quantifies each token's
probabilistic contribution to the model's prediction.

Reference: Section 4 — *KL Divergence for Information Contribution*; Eq. (4).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from heta.utils import (
    mask_token_embedding,
)

logger = logging.getLogger(__name__)


class KLInformationImpact:
    """Compute per-token KL-based information contribution I(x_i → x_T).

    For each context token x_i, we mask it and measure the KL divergence
    between the original and perturbed target distributions:

        I(x_i → x_T) = D_{KL}( P_orig(· | x_{<T}) ‖ P_masked^{(i)}(· | x_{<T}) )

    Higher values indicate that masking the token substantially changes
    the model's prediction, implying a larger information contribution.
    """

    def __init__(self, mask_scheme: str = "zero") -> None:
        """
        Args:
            mask_scheme: Masking strategy — ``"zero"`` | ``"mean"`` | ``"unk"``.
        """
        self.mask_scheme = mask_scheme

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(
        self,
        model: PreTrainedModel,
        embeddings: torch.Tensor,
        target_pos: int,
        original_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token KL information impact.

        Args:
            model:           The language model.
            embeddings:      Input embeddings, shape ``[1, T, d]``.
            target_pos:      Position of the target token (0-indexed).
                             Logits are read at ``target_pos - 1`` because
                             autoregressive LMs predict the next token.
            original_logits: Pre-computed logits (optional; avoids an extra
                             forward pass when available).

        Returns:
            Tensor of shape ``[num_context]`` with non-negative KL scores,
            where ``num_context = target_pos``.
        """
        device = embeddings.device
        num_context = target_pos
        logit_pos = target_pos - 1  # autoregressive: logits[t] predicts token t+1

        # Original distribution at the prediction position
        if original_logits is None:
            original_logits = model(
                inputs_embeds=embeddings, use_cache=False
            ).logits
        p_orig = F.softmax(original_logits[0, logit_pos, :], dim=-1)

        kl_scores = torch.zeros(num_context, device=device, dtype=torch.float32)

        for i in range(num_context):
            masked_embeds = mask_token_embedding(
                embeddings, token_idx=i, scheme=self.mask_scheme
            )
            masked_logits = model(
                inputs_embeds=masked_embeds,
                use_cache=False,
            ).logits
            p_masked = F.softmax(masked_logits[0, logit_pos, :], dim=-1)

            # KL divergence: D_KL(P_orig || P_masked)
            kl = self._kl_divergence(p_orig, p_masked)
            kl_scores[i] = kl

        return kl_scores

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute D_KL(p ‖ q) with numerical clamping.

        Args:
            p: Reference distribution (1-D probability vector).
            q: Comparison distribution (same shape as *p*).

        Returns:
            Non-negative scalar tensor.
        """
        eps = 1e-10
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        kl = (p * (p.log() - q.log())).sum()
        return kl.clamp(min=0.0)
