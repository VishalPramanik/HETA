#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
Evaluate HETA on the curated NarrativeQA ⊕ SciQ attribution dataset.

Loads the curated dataset, runs HETA attribution for each example, and reports
the Dependent Sentence Attribution (DSA) metric along with per-example
diagnostics.

Usage:
    # Quick check (first 5 examples from the 100-unique set)
    python scripts/evaluate_curated.py --model gpt2 --max-examples 5

    # Full evaluation on the 100-unique set
    python scripts/evaluate_curated.py \
        --model gpt2 \
        --dataset data/heta_qa_dataset_100_unique.json \
        --output results/curated_gpt2.json

    # Full evaluation on the 2000-example set with GPU
    python scripts/evaluate_curated.py \
        --model EleutherAI/gpt-j-6b \
        --dataset data/heta_qa_dataset_2000.json \
        --device cuda \
        --output results/curated_gptj.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heta import HETA
from heta.metrics import DSA, SoftNC, SoftNS, f1_alignment
from heta.utils import HETAConfig
from heta.visualization import print_attribution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("heta.eval_curated")


# ======================================================================
# Dataset loader
# ======================================================================

def load_curated_dataset(path: str, max_examples: int = 0) -> List[Dict]:
    """Load the curated NarrativeQA ⊕ SciQ JSON dataset.

    Args:
        path:         Path to the JSON file.
        max_examples: If > 0, truncate to this many examples.

    Returns:
        List of example dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]
    if max_examples > 0:
        examples = examples[:max_examples]

    logger.info(
        "Loaded %d examples from %s (%d total in file)",
        len(examples), path, data["num_examples"],
    )
    return examples


def locate_token_indices(
    tokenizer,
    input_text: str,
    narrative_segment: str,
    sciq_support_segment: str,
    important_words: List[str],
) -> Tuple[List[int], List[int], List[int]]:
    """Map word-level annotations to subword token indices.

    Returns:
        (narrative_indices, sciq_indices, important_indices) — each a list
        of 0-indexed token positions in the tokenised input.
    """
    encoding = tokenizer(input_text, add_special_tokens=True)
    token_ids = encoding.input_ids
    tokens = [tokenizer.decode([tid]).lower().strip() for tid in token_ids]

    # Find the approximate boundary between segments using the <s> separator
    full_lower = input_text.lower()
    sep_positions = [m.start() for m in re.finditer(r"<s>", full_lower)]

    # Character offset of each token (approximate)
    narrative_indices = []
    sciq_indices = []
    important_indices = []
    important_set = {w.lower() for w in important_words}

    # Simple heuristic: tokens before first <s> are narrative, between first
    # and second <s> are SciQ support, after second <s> are question.
    narrative_lower = narrative_segment.lower()
    sciq_lower = sciq_support_segment.lower()

    for idx, tok_str in enumerate(tokens):
        clean = tok_str.strip().lower()
        if not clean or clean in {"<s>", "<", "s", ">", "question", ":", ""}:
            continue

        # Check if token belongs to narrative vs SciQ
        if clean in narrative_lower and clean not in sciq_lower:
            narrative_indices.append(idx)
        elif clean in sciq_lower:
            sciq_indices.append(idx)
            if clean in important_set:
                important_indices.append(idx)

    return narrative_indices, sciq_indices, important_indices


# ======================================================================
# Evaluation loop
# ======================================================================

def evaluate(args: argparse.Namespace) -> None:
    config = HETAConfig(
        beta=args.beta,
        gamma=args.gamma,
        num_hvp_samples=args.hvp_samples,
        mask_scheme=args.mask_scheme,
        device=args.device,
    )

    logger.info("Loading model: %s", args.model)
    attributor = HETA.from_pretrained(args.model, config=config)
    logger.info("Model loaded on %s", attributor.device)

    examples = load_curated_dataset(args.dataset, args.max_examples)

    results = []
    total_dsa = 0.0
    total_f1 = 0.0
    num_valid = 0

    for idx, ex in enumerate(examples):
        t0 = time.time()

        input_text = ex["input_text"]
        important_words = ex["important_words"]
        answer = ex["answer"]

        try:
            result = attributor.attribute(input_text)
        except Exception as e:
            logger.warning("Example %d failed: %s", idx, e)
            continue

        elapsed = time.time() - t0

        # Map annotations to token indices
        narr_idxs, sciq_idxs, imp_idxs = locate_token_indices(
            attributor.tokenizer,
            input_text,
            ex["narrative_segment"],
            ex["sciq_support_segment"],
            important_words,
        )

        # Clamp indices to attribution range
        n_ctx = len(result.scores)
        narr_idxs = [i for i in narr_idxs if i < n_ctx]
        sciq_idxs = [i for i in sciq_idxs if i < n_ctx]
        imp_idxs = [i for i in imp_idxs if i < n_ctx]

        # Compute DSA
        dsa = DSA.evaluate(result.scores, sciq_idxs, narr_idxs)

        # Compute F1 alignment
        f1 = f1_alignment(result.scores, imp_idxs) if imp_idxs else 0.0

        total_dsa += dsa
        total_f1 += f1
        num_valid += 1

        entry = {
            "id": ex["id"],
            "answer": answer,
            "target_token": result.target_token,
            "dsa": round(dsa, 4),
            "f1": round(f1, 4),
            "time_sec": round(elapsed, 3),
            "top5": [(t, round(s, 4)) for t, s in result.topk(5)],
            "num_narrative_tokens": len(narr_idxs),
            "num_sciq_tokens": len(sciq_idxs),
            "num_important_tokens": len(imp_idxs),
        }
        results.append(entry)

        if idx < 3 or args.verbose:
            print_attribution(
                result.tokens, result.scores, result.target_token, top_k=8
            )

        if (idx + 1) % 10 == 0 or idx == len(examples) - 1:
            logger.info(
                "Example %d/%d | DSA=%.4f | F1=%.4f | %.2fs",
                idx + 1, len(examples), dsa, f1, elapsed,
            )

    # Summary
    if num_valid > 0:
        avg_dsa = total_dsa / num_valid
        avg_f1 = total_f1 / num_valid
    else:
        avg_dsa = avg_f1 = 0.0

    logger.info("=" * 60)
    logger.info("  Model:        %s", args.model)
    logger.info("  Examples:     %d", num_valid)
    logger.info("  Avg DSA:      %.4f", avg_dsa)
    logger.info("  Avg F1:       %.4f", avg_f1)
    logger.info("=" * 60)

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "config": {"beta": args.beta, "gamma": args.gamma},
            "num_examples": num_valid,
            "avg_dsa": round(avg_dsa, 4),
            "avg_f1": round(avg_f1, 4),
            "per_example": results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HETA — Curated Dataset Evaluation (DSA metric)"
    )
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model name or path.")
    p.add_argument("--dataset", default="data/heta_qa_dataset_100_unique.json",
                   help="Path to the curated JSON dataset.")
    p.add_argument("--max-examples", type=int, default=0,
                   help="Evaluate only the first N examples (0 = all).")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--hvp-samples", type=int, default=10)
    p.add_argument("--mask-scheme", default="zero",
                   choices=["zero", "mean", "unk"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None,
                   help="Output JSON path for results.")
    p.add_argument("--verbose", action="store_true",
                   help="Print attribution for every example.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
