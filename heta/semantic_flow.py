# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
Semantic Transition Influence — Component (1) of HETA.

Traces attention-weighted value flow through transformer layers, restricted to
causal paths that terminate at the target prediction position.  The resulting
vector M_T ∈ ℝ^{T}_{≥0} is simplex-normalized and acts as a *causal gate*
over input tokens (see Section 4 of the paper).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SemanticTransitionInfluence:
    """Compute the target-conditioned semantic transition vector M_T.

    For each layer *l* and head *h*, we form a target-conditioned attention
    rollout Φ^{(l,h)}_{(i→T)} that aggregates only causal paths ending at the
    prediction position.  The transition influence is (paper Section 4):

        M_T[i] = (1/Z) Σ_{l,h} Φ^{(l,h)}_{(i→T)} · ‖V_i^{(l,h)} W_O^{(l,h)}‖₁

    where V_i^{(l,h)} is the per-token value vector (activation-dependent),
    W_O^{(l,h)} is the output projection, and Z normalises to a simplex.

    Reference: Section 4 — *Semantic Flow for Causal Token Influence*;
               Kobayashi et al. (2020) for value-norm weighting.
    """

    def __init__(self, layer_subset: int = 0) -> None:
        self.layer_subset = layer_subset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(
        self,
        attentions: Tuple[torch.Tensor, ...],
        model: torch.nn.Module,
        rollout_target: int,
        num_context: int,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """Compute M_T for a single sequence.

        Args:
            attentions:     Tuple of per-layer attention tensors, each of shape
                            ``[1, H, T, T]`` (already causally masked).
            model:          The HuggingFace model (used to read value/output
                            projection weights).
            rollout_target: Position to trace rollout paths to (= logit_pos).
            num_context:    Number of context tokens to score.
            hidden_states:  Optional tuple of per-layer hidden states, each
                            ``[1, T, d]``.  When provided, value norms are
                            computed per-token (paper-faithful).  Without them,
                            falls back to weight-only norms.

        Returns:
            ``M_T`` — tensor of shape ``[num_context]``, simplex-normalized.
        """
        num_layers = len(attentions)
        start_layer = (
            max(0, num_layers - self.layer_subset) if self.layer_subset > 0
            else 0
        )
        seq_len = attentions[0].shape[-1]
        device = attentions[0].device

        # Accumulate un-normalized transition scores
        mt = torch.zeros(seq_len, device=device, dtype=torch.float32)

        # Compute rollout across selected layers
        rollout = self._compute_rollout(attentions, start_layer, rollout_target)

        # Compute per-token value-projection norms ‖V_i^{(l,h)} W_O^{(l,h)}‖₁
        value_norms_per_layer = self._compute_value_norms(
            model, attentions, hidden_states, start_layer, num_layers,
            seq_len, device,
        )

        # M_T[i] = (1/Z) Σ_{l,h} Φ(i→T) · ‖V_i W_O‖₁
        for li, layer_idx in enumerate(range(start_layer, num_layers)):
            num_heads = attentions[layer_idx].shape[1]
            for head_idx in range(num_heads):
                phi = rollout[li, head_idx]          # [seq_len]
                v_norms = value_norms_per_layer[li][head_idx]  # [seq_len]
                mt[:num_context] += phi[:num_context] * v_norms[:num_context]

        # Simplex normalisation
        mt_ctx = mt[:num_context].clamp(min=0.0)
        total = mt_ctx.sum()
        if total > 0:
            mt_ctx = mt_ctx / total
        else:
            mt_ctx = torch.ones(num_context, device=device) / num_context
        return mt_ctx

    # ------------------------------------------------------------------
    # Attention rollout  (Abnar & Zuidema, 2020)
    # ------------------------------------------------------------------

    def _compute_rollout(
        self,
        attentions: Tuple[torch.Tensor, ...],
        start_layer: int,
        rollout_target: int,
    ) -> torch.Tensor:
        """Target-conditioned attention rollout.

        Layer-by-layer: R ← (0.5·A + 0.5·I) @ R,  then read row
        ``rollout_target``.

        Returns:
            ``[num_selected_layers, H, seq_len]``.
        """
        selected = list(range(start_layer, len(attentions)))
        seq_len = attentions[0].shape[-1]
        num_heads = attentions[0].shape[1]
        device = attentions[0].device

        out = torch.zeros(
            len(selected), num_heads, seq_len,
            device=device, dtype=torch.float32,
        )
        eye = torch.eye(seq_len, device=device, dtype=torch.float32)

        for head_idx in range(num_heads):
            R = eye.clone()
            for li, layer_idx in enumerate(selected):
                A = attentions[layer_idx][0, head_idx].float()
                A_hat = 0.5 * A + 0.5 * eye
                A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                R = A_hat @ R
                out[li, head_idx, :] = R[rollout_target, :]
        return out

    # ------------------------------------------------------------------
    # Per-token value-projection norms  (Kobayashi et al. 2020)
    # ------------------------------------------------------------------

    def _compute_value_norms(
        self,
        model: torch.nn.Module,
        attentions: Tuple[torch.Tensor, ...],
        hidden_states: Optional[Tuple[torch.Tensor, ...]],
        start_layer: int,
        num_layers: int,
        seq_len: int,
        device: torch.device,
    ) -> List[List[torch.Tensor]]:
        """Compute ‖V_i^{(l,h)} W_O^{(l,h)}‖₁ per token per head per layer.

        When ``hidden_states`` is available, the value vectors are computed
        per-token:  V_i = h_i @ W_V  (activation-dependent, paper-faithful).
        Otherwise falls back to weight-only norms.

        Returns:
            ``result[layer_offset][head]`` → tensor of shape ``[seq_len]``.
        """
        result = []
        for layer_idx in range(start_layer, num_layers):
            num_heads = attentions[layer_idx].shape[1]
            # hidden_states[layer_idx] is the input to this layer
            hs = hidden_states[layer_idx] if hidden_states is not None else None
            norms = self._value_norms_for_layer(
                model, layer_idx, num_heads, seq_len, device, hs
            )
            result.append(norms)
        return result

    @staticmethod
    def _value_norms_for_layer(
        model: torch.nn.Module,
        layer_idx: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
        hidden_state: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Per-token ‖V_i^{(l,h)} W_O^{(l,h)}‖₁ for one layer.

        Supports GPT-2/GPT-J (c_attn + c_proj / q/k/v_proj + out_proj)
        and LLaMA/Qwen/Phi (v_proj + o_proj).
        Falls back to uniform 1.0 when architecture is unrecognised.
        """
        ones = [torch.ones(seq_len, device=device) for _ in range(num_heads)]

        try:
            # --- Locate the transformer block and attention module ---
            block = None
            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                block = model.transformer.h[layer_idx]
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                block = model.model.layers[layer_idx]
            if block is None:
                return ones

            attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
            if attn is None:
                return ones

            # --- Extract W_V and W_O weight matrices ---
            W_V, W_O = None, None

            # LLaMA / Qwen / Phi: separate v_proj and o_proj
            if hasattr(attn, "v_proj") and hasattr(attn, "o_proj"):
                W_V = attn.v_proj.weight.detach().float()   # [d_v, d]
                W_O = attn.o_proj.weight.detach().float()   # [d, d_v]
            # GPT-2: fused c_attn (QKV) and c_proj
            elif hasattr(attn, "c_attn") and hasattr(attn, "c_proj"):
                qkv_w = attn.c_attn.weight.detach().float()  # [d, 3d]
                d = qkv_w.shape[0]
                W_V = qkv_w[:, 2 * d:].T    # [d, d] → transpose to [d, d]
                W_O = attn.c_proj.weight.detach().float().T  # [d, d]
            # GPT-J: separate q/k/v_proj + out_proj
            elif hasattr(attn, "v_proj") and hasattr(attn, "out_proj"):
                W_V = attn.v_proj.weight.detach().float()
                W_O = attn.out_proj.weight.detach().float()

            if W_V is None or W_O is None:
                return ones

            d_model = W_O.shape[0]
            head_dim = W_V.shape[0] // num_heads

            # --- Compute per-token, per-head value-projection norms ---
            # V_i^{(l,h)} = h_i @ W_V^{(h)T}  (shape [head_dim])
            # output_i    = V_i^{(l,h)} @ W_O^{(h)T}  (shape [d_model])
            # norm_i      = ‖output_i‖₁

            if hidden_state is not None:
                # hidden_state: [1, T, d_model]
                H = hidden_state[0].detach().float()  # [T, d_model]
                head_norms = []
                for h in range(num_heads):
                    s, e = h * head_dim, (h + 1) * head_dim
                    W_V_h = W_V[s:e, :]       # [head_dim, d_model]
                    W_O_h = W_O[:, s:e]        # [d_model, head_dim]
                    # V_all = H @ W_V_h^T → [T, head_dim]
                    V_all = H @ W_V_h.T
                    # projected = V_all @ W_O_h^T → [T, d_model]
                    projected = V_all @ W_O_h.T
                    # per-token ℓ₁ norm
                    norms = projected.abs().sum(dim=-1)  # [T]
                    # Pad if seq_len > T (shouldn't happen, but be safe)
                    if norms.shape[0] < seq_len:
                        norms = F.pad(norms, (0, seq_len - norms.shape[0]), value=1.0)
                    head_norms.append(norms[:seq_len])
                return head_norms
            else:
                # Fallback: weight-only norm (same for all tokens)
                head_norms = []
                for h in range(num_heads):
                    s, e = h * head_dim, (h + 1) * head_dim
                    W_V_h = W_V[s:e, :]
                    W_O_h = W_O[:, s:e]
                    norm_scalar = (W_O_h @ W_V_h).norm(p=1).item()
                    head_norms.append(
                        torch.full((seq_len,), norm_scalar, device=device)
                    )
                return head_norms

        except Exception as exc:
            logger.debug("Value-norm extraction failed for layer %d: %s", layer_idx, exc)
            return ones
