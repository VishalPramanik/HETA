# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
HETA: Hessian-Enhanced Token Attribution.

Unified attribution pipeline that combines:
  (1) Semantic Transition Influence  —  causal gate M_T[i]
  (2) Hessian-Based Sensitivity      —  curvature score S_i^{(T)}
  (3) KL Information Impact           —  distributional shift I(x_i → x_T)

into the final target-conditioned attribution (Eq. 5):

    Attr(x_i → x_T) = M_T[i] · ( β · S_i^{(T)} + γ · I(x_i → x_T) )

Reference: Section 4 of the paper; Algorithm 1 (Appendix A3).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from heta.hessian_sensitivity import HessianSensitivity
from heta.kl_divergence import KLInformationImpact
from heta.semantic_flow import SemanticTransitionInfluence
from heta.utils import (
    AttributionResult,
    HETAConfig,
    build_windows,
    forward_with_embeddings,
    get_embeddings,
    get_target_log_prob,
    resolve_device,
)

logger = logging.getLogger(__name__)


class HETA:
    """Hessian-Enhanced Token Attribution for decoder-only LMs.

    Example::

        from heta import HETA, HETAConfig

        cfg = HETAConfig(beta=0.5, gamma=0.5, device="cuda")
        attributor = HETA.from_pretrained("gpt2", config=cfg)
        result = attributor.attribute("The capital of France is Paris")
        for tok, score in result.topk(5):
            print(f"  {tok:>12s}  {score:.4f}")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[HETAConfig] = None,
    ) -> None:
        self.config = config or HETAConfig()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

        # Sub-modules
        self._flow = SemanticTransitionInfluence(
            layer_subset=self.config.layer_subset,
        )
        self._hess = HessianSensitivity(
            num_samples=self.config.num_hvp_samples,
            low_rank=self.config.low_rank,
            use_fisher=self.config.use_fisher,
        )
        self._kl = KLInformationImpact(mask_scheme=self.config.mask_scheme)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[HETAConfig] = None,
        **model_kwargs,
    ) -> "HETA":
        """Load a HuggingFace causal LM and wrap it in the HETA pipeline.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            config:             Optional ``HETAConfig`` instance.
            **model_kwargs:     Extra keyword arguments forwarded to
                                ``AutoModelForCausalLM.from_pretrained``.

        Returns:
            A ready-to-use ``HETA`` instance.
        """
        cfg = config or HETAConfig()
        device = resolve_device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=cfg.dtype,
            output_attentions=True,
            **model_kwargs,
        )
        return cls(model, tokenizer, config=cfg)

    # ------------------------------------------------------------------
    # Main attribution entry point
    # ------------------------------------------------------------------

    def attribute(
        self,
        text: str,
        target_pos: Optional[int] = None,
        target_token_id: Optional[int] = None,
    ) -> AttributionResult:
        """Compute HETA attributions for a single input text.

        By default, the target is the *last* token in the sequence and
        attribution is computed for all preceding context tokens.

        Args:
            text:            Input text string.
            target_pos:      Override target position (0-indexed).  If ``None``,
                             uses the last token position.
            target_token_id: Override target token ID.  If ``None``, uses the
                             token at ``target_pos``.

        Returns:
            An ``AttributionResult`` with per-token scores and metadata.
        """
        # Tokenize
        encoding = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoding.input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            raise ValueError(
                "Input must contain at least 2 tokens (1 context + 1 target)."
            )

        # Determine target
        if target_pos is None:
            target_pos = seq_len - 1
        if target_token_id is None:
            target_token_id = input_ids[0, target_pos].item()

        num_context = target_pos  # tokens 0 .. target_pos-1

        # Token strings for display
        all_tokens = [
            self.tokenizer.decode([input_ids[0, j].item()])
            for j in range(seq_len)
        ]

        # Decide between windowed and full attribution
        if (
            self.config.window_size > 0
            and num_context > self.config.window_size
        ):
            result = self._attribute_windowed(
                input_ids, target_pos, target_token_id, all_tokens
            )
        else:
            result = self._attribute_full(
                input_ids, target_pos, target_token_id, all_tokens
            )

        return result

    # ------------------------------------------------------------------
    # Full (non-windowed) attribution
    # ------------------------------------------------------------------

    def _attribute_full(
        self,
        input_ids: torch.Tensor,
        target_pos: int,
        target_token_id: int,
        all_tokens: List[str],
    ) -> AttributionResult:
        """Run full HETA on the entire context.

        Key indexing: in autoregressive LMs, logits[0, t, :] predicts the
        token at position t+1.  To predict x_{target_pos}, we read logits at
        ``logit_pos = target_pos - 1``.  The semantic rollout also targets
        ``logit_pos`` because that is the representation that produces the
        prediction.  Context tokens are 0 .. target_pos-1 (= logit_pos).
        """
        num_context = target_pos
        logit_pos = target_pos - 1  # position whose logits predict x_{target_pos}

        # ---------- Forward pass with embeddings ----------
        embeddings = get_embeddings(self.model, input_ids)
        logits, attentions, hidden_states = forward_with_embeddings(
            self.model, embeddings
        )

        # ---------- (1) Semantic transition influence ----------
        # Trace attention-value paths terminating at logit_pos (the position
        # that generates the prediction for x_T).
        # Pass hidden_states so M_T can compute per-token ‖V_i W_O‖₁ norms.
        mt = self._flow.compute(
            attentions, self.model, logit_pos, num_context,
            hidden_states=hidden_states,
        )

        # ---------- (2) Hessian-based sensitivity ----------
        # get_target_log_prob internally reads logits[0, target_pos-1, :]
        log_prob = get_target_log_prob(logits, target_token_id, target_pos)
        si = self._hess.compute(log_prob, embeddings, num_context)

        # ---------- (3) KL information impact ----------
        # Detach embeddings for the KL sweep (no gradient needed)
        embeds_detached = embeddings.detach()
        ii = self._kl.compute(
            self.model, embeds_detached, target_pos,
            original_logits=logits.detach(),
        )

        # ---------- Final attribution (Eq. 5) ----------
        scores = mt * (self.config.beta * si + self.config.gamma * ii)

        return AttributionResult(
            scores=scores,
            semantic_flow=mt,
            hessian_sensitivity=si,
            kl_divergence=ii,
            tokens=all_tokens[:target_pos],
            target_token=all_tokens[target_pos],
            target_position=target_pos,
        )

    # ------------------------------------------------------------------
    # Windowed attribution for long contexts
    # ------------------------------------------------------------------

    def _attribute_windowed(
        self,
        input_ids: torch.Tensor,
        target_pos: int,
        target_token_id: int,
        all_tokens: List[str],
    ) -> AttributionResult:
        """Run HETA with sliding-window accumulation (Algorithm 1, lines 4–27).

        Windows of size ``config.window_size`` with ``config.window_overlap``
        fractional overlap cover the full context.  Per-window scores are
        averaged for tokens appearing in multiple windows.
        """
        num_context = target_pos
        device = self.device
        W = self.config.window_size
        overlap = self.config.window_overlap

        windows = build_windows(num_context, W, overlap)

        mt_accum = torch.zeros(num_context, device=device)
        si_accum = torch.zeros(num_context, device=device)
        ii_accum = torch.zeros(num_context, device=device)
        counts = torch.zeros(num_context, device=device)

        for start, end in windows:
            # Build windowed input: context[start:end] + target token
            window_ids = torch.cat(
                [
                    input_ids[:, start:end],
                    input_ids[:, target_pos : target_pos + 1],
                ],
                dim=1,
            )
            win_target_pos = end - start  # target token position in the window
            win_logit_pos = win_target_pos - 1  # logits position
            win_num_context = win_target_pos  # context tokens: 0..win_target_pos-1

            embeddings = get_embeddings(self.model, window_ids)
            logits, attentions, hidden_states = forward_with_embeddings(
                self.model, embeddings
            )

            # (1) Semantic flow — trace to win_logit_pos
            mt_win = self._flow.compute(
                attentions, self.model, win_logit_pos, win_num_context,
                hidden_states=hidden_states,
            )
            # (2) Hessian sensitivity
            log_prob = get_target_log_prob(logits, target_token_id, win_target_pos)
            si_win = self._hess.compute(log_prob, embeddings, win_num_context)
            # (3) KL
            embeds_d = embeddings.detach()
            ii_win = self._kl.compute(
                self.model, embeds_d, win_target_pos,
                original_logits=logits.detach(),
            )

            # Map back to global indices
            for local_i, global_i in enumerate(range(start, end)):
                mt_accum[global_i] += mt_win[local_i]
                si_accum[global_i] += si_win[local_i]
                ii_accum[global_i] += ii_win[local_i]
                counts[global_i] += 1.0

        # Average overlapping windows
        counts = counts.clamp(min=1.0)
        mt = mt_accum / counts
        si = si_accum / counts
        ii = ii_accum / counts

        # Re-normalize semantic flow to simplex
        mt_sum = mt.sum()
        if mt_sum > 0:
            mt = mt / mt_sum

        scores = mt * (self.config.beta * si + self.config.gamma * ii)

        return AttributionResult(
            scores=scores,
            semantic_flow=mt,
            hessian_sensitivity=si,
            kl_divergence=ii,
            tokens=all_tokens[:target_pos],
            target_token=all_tokens[target_pos],
            target_position=target_pos,
        )

    # ------------------------------------------------------------------
    # Convenience: batch attribution
    # ------------------------------------------------------------------

    def attribute_batch(
        self,
        texts: List[str],
        target_positions: Optional[List[int]] = None,
    ) -> List[AttributionResult]:
        """Attribute a list of texts sequentially.

        Args:
            texts:             List of input strings.
            target_positions:  Per-text target positions (optional).

        Returns:
            List of ``AttributionResult`` instances.
        """
        results = []
        for idx, text in enumerate(texts):
            tp = target_positions[idx] if target_positions else None
            results.append(self.attribute(text, target_pos=tp))
        return results
