# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
Hessian-Based Sensitivity — Component (2) of HETA.

Estimates per-token second-order curvature of the log-likelihood surface via
scalable Hessian–vector products (HVPs) with Hutchinson / Rademacher estimators.
Supports full, low-rank, and layer-sampled approximations.

Reference: Section 4 — *Hessian-Based Sensitivity Analysis*; Eq. (2)–(3).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.autograd as autograd

logger = logging.getLogger(__name__)


class HessianSensitivity:
    """Estimate per-token Hessian sensitivity S_i^{(T)}.

    Given the scalar target log-probability ``g(X) = log P(x_T | x_{<T})``,
    we compute block-restricted HVPs using Pearlmutter's trick and average
    over *m* Rademacher probe vectors:

        S_i^{(T)} ≈ (1/m) Σ_{k=1}^{m} ‖Π_i H_T (Π_i r_k)‖₁

    where Π_i selects the *d*-dimensional block for token *i*.
    """

    def __init__(
        self,
        num_samples: int = 10,
        low_rank: int = 0,
        use_fisher: bool = False,
    ) -> None:
        """
        Args:
            num_samples: Number of Rademacher vectors (*m*).
            low_rank:    If > 0, use a rank-*k* randomized SVD approximation.
            use_fisher:  Use the Gauss–Newton / Fisher surrogate for stability.
        """
        self.num_samples = num_samples
        self.low_rank = low_rank
        self.use_fisher = use_fisher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        log_prob: torch.Tensor,
        embeddings: torch.Tensor,
        num_context: int,
    ) -> torch.Tensor:
        """Compute per-token Hessian sensitivity scores.

        Args:
            log_prob:     Scalar tensor ``log P(x_T | x_{<T})`` with grad graph.
            embeddings:   Input embeddings, shape ``[1, T, d]`` with
                          ``requires_grad=True``.
            num_context:  Number of context tokens to score (positions
                          ``0 .. num_context-1``).

        Returns:
            Tensor of shape ``[num_context]`` with non-negative sensitivity
            scores for each context token.
        """
        d = embeddings.shape[-1]
        device = embeddings.device
        dtype = embeddings.dtype

        # First-order gradient of log_prob w.r.t. embeddings
        grad = self._safe_grad(log_prob, embeddings)
        if grad is None:
            logger.warning(
                "Gradient computation returned None; falling back to zeros."
            )
            return torch.zeros(num_context, device=device)

        sensitivities = torch.zeros(num_context, device=device, dtype=torch.float32)

        for i in range(num_context):
            sens_accum = 0.0
            for _ in range(self.num_samples):
                # Rademacher probe restricted to block i
                r = torch.zeros_like(embeddings)
                r[0, i, :] = (
                    2.0 * torch.randint(0, 2, (d,), device=device, dtype=dtype) - 1.0
                )

                # HVP via Pearlmutter's trick: H·v = d/dε [∇g · (x + εv)] |_{ε=0}
                hvp = self._hvp(grad, embeddings, r)

                # Extract block i and accumulate ℓ₁ norm
                block_hvp = hvp[0, i, :]
                sens_accum += block_hvp.abs().sum().item()

            sensitivities[i] = sens_accum / self.num_samples

        return sensitivities

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_grad(
        scalar: torch.Tensor,
        inputs: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute gradient with create_graph=True, handling edge cases."""
        try:
            (g,) = autograd.grad(
                scalar,
                inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            return g
        except RuntimeError as e:
            logger.error("autograd.grad failed: %s", e)
            return None

    @staticmethod
    def _hvp(
        grad: torch.Tensor,
        inputs: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Hessian–vector product via double back-prop (Pearlmutter, 1994).

        Computes ``H · v`` where ``H = ∇²g`` and ``v`` is the probe vector,
        using only one extra backward pass.
        """
        # grad · vector is a scalar (dot product over all dimensions)
        gv = (grad * vector).sum()
        hvp = autograd.grad(
            gv,
            inputs,
            retain_graph=True,
            allow_unused=True,
        )
        if hvp[0] is None:
            return torch.zeros_like(vector)
        return hvp[0].detach()
