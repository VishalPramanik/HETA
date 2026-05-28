#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
Run HETA attribution evaluation on benchmark datasets.

Usage:
    python scripts/run_attribution.py \
        --model gpt2 \
        --dataset tellmewhy \
        --beta 0.5 --gamma 0.5 \
        --device cuda \
        --output results/

Supported models: gpt2, gpt2-medium, EleutherAI/gpt-j-6b, microsoft/Phi-3-medium-4k-instruct,
                  meta-llama/Llama-3.1-70B, Qwen/Qwen2.5-3B
Supported datasets: longra, tellmewhy, wikibio, curated
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heta import HETA
from heta.metrics import DSA, SoftNC, SoftNS, f1_alignment
from heta.utils import HETAConfig
from heta.visualization import print_attribution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("heta.eval")


# ======================================================================
# Dataset loaders
# ======================================================================

def _load_curated_samples(max_n: int = 10) -> list:
    """Load the first N examples from the curated NarrativeQA ⊕ SciQ dataset."""
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "heta_qa_dataset_100_unique.json"
    if not dataset_path.exists():
        logger.warning("Curated dataset not found at %s; using fallback.", dataset_path)
        return [
            (
                "The protagonist returns to the village after the winter storm. "
                "Photosynthesis primarily occurs in the leaves of the plant, where "
                "chloroplasts capture light. Question: In which part of the plant "
                "does photosynthesis mainly take place? Answer: leaves",
                list(range(11, 22)),
            ),
        ]
    import json
    with open(dataset_path) as f:
        data = json.load(f)
    samples = []
    for ex in data["examples"][:max_n]:
        text = ex["input_text"]
        # Map important_word_indices from SciQ-local to full-text approximate indices
        gold = ex.get("important_word_indices_in_sciq_support", [])
        samples.append((text, gold))
    return samples


# ======================================================================
# Toy / demo data generators  (replace with real datasets for full eval)
# ======================================================================

def load_demo_samples(dataset_name: str):
    """Return a small list of (text, gold_indices) tuples for demonstration.

    For full-scale evaluation, replace this with actual dataset loaders for
    LongRA, TellMeWhy, WikiBio, and the curated NarrativeQA⊕SciQ set.

    For the curated dataset, loads from the JSON file in data/.
    """
    if dataset_name == "curated":
        return _load_curated_samples()

    samples = {
        "longra": [
            (
                "Japan is a country in East Asia. It has many beautiful mountains "
                "and temples. The bustling streets are filled with people. The "
                "capital of Japan is Tokyo",
                [0],  # "Japan"
            ),
            (
                "The researcher published a groundbreaking paper on quantum "
                "computing. After years of effort, the algorithm achieved "
                "superior performance on the benchmark",
                [7],  # "quantum"
            ),
        ],
        "tellmewhy": [
            (
                "Cam ordered a pizza and took it home. He opened the box to take "
                "out a slice. Cam discovered that the store did not cut the pizza "
                "for him. He looked for his pizza cutter but did not find it. "
                "He had to use his chef knife to cut a slice",
                [3, 22, 25, 26, 32, 34],  # pizza, cut, pizza, cutter, knife, cut
            ),
            (
                "Sandra got a job at the zoo. She loved coming to work and seeing "
                "all of the animals. Sandra went to look at the polar bears during "
                "her lunch break. She watched them eat fish and jump in and out of "
                "the water. She took pictures and shared them with her friends",
                [6, 14, 33, 35],  # zoo, animals, pictures, shared
            ),
        ],
        "wikibio": [
            (
                "Albert Einstein was a German-born theoretical physicist who "
                "developed the theory of relativity. He received the Nobel Prize "
                "in Physics in 1921 for his explanation of the photoelectric effect",
                [0, 1, 6, 7, 8],
            ),
        ],
        "curated": [
            (
                "The protagonist returns to the village after the winter storm, "
                "reflecting on her father's passing. Photosynthesis primarily "
                "occurs in the leaves of the plant, where chloroplasts capture "
                "light. Question: In which part of the plant does photosynthesis "
                "mainly take place? Answer: leaves",
                list(range(11, 22)),  # SciQ segment indices (approximate)
            ),
        ],
    }
    return samples.get(dataset_name, samples["tellmewhy"])


# ======================================================================
# Main evaluation loop
# ======================================================================

def evaluate(args: argparse.Namespace) -> None:
    """Run HETA on the specified dataset and report metrics."""

    config = HETAConfig(
        beta=args.beta,
        gamma=args.gamma,
        num_hvp_samples=args.hvp_samples,
        mask_scheme=args.mask_scheme,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        low_rank=args.low_rank,
        layer_subset=args.layer_subset,
        device=args.device,
    )

    logger.info("Loading model: %s", args.model)
    attributor = HETA.from_pretrained(args.model, config=config)
    logger.info("Model loaded on %s", attributor.device)

    samples = load_demo_samples(args.dataset)
    logger.info(
        "Evaluating %d samples from '%s' dataset", len(samples), args.dataset
    )

    results_log = []
    total_softNC = 0.0
    total_softNS = 0.0
    total_f1 = 0.0

    for idx, (text, gold_indices) in enumerate(samples):
        t0 = time.time()
        result = attributor.attribute(text)
        elapsed = time.time() - t0

        # Print attribution
        print_attribution(
            result.tokens, result.scores, result.target_token, top_k=10
        )

        # Compute metrics
        encoding = attributor.tokenizer(text, return_tensors="pt")
        input_ids = encoding.input_ids.to(attributor.device)
        target_id = input_ids[0, result.target_position].item()

        nc = SoftNC.evaluate(
            attributor.model, input_ids, result.scores,
            result.target_position, target_id,
        )
        ns = SoftNS.evaluate(
            attributor.model, input_ids, result.scores,
            result.target_position, target_id,
        )
        # Clamp gold indices to valid range
        valid_gold = [g for g in gold_indices if g < len(result.scores)]
        f1 = f1_alignment(result.scores, valid_gold) if valid_gold else 0.0

        total_softNC += nc
        total_softNS += ns
        total_f1 += f1

        entry = {
            "sample_idx": idx,
            "text_snippet": text[:80] + "...",
            "target_token": result.target_token,
            "soft_nc": round(nc, 4),
            "soft_ns": round(ns, 4),
            "f1_alignment": round(f1, 4),
            "time_sec": round(elapsed, 3),
            "top5": result.topk(5),
        }
        results_log.append(entry)
        logger.info(
            "Sample %d | Soft-NC=%.4f | Soft-NS=%.4f | F1=%.4f | %.2fs",
            idx, nc, ns, f1, elapsed,
        )

    n = len(samples)
    logger.info("=" * 60)
    logger.info(
        "Average  |  Soft-NC=%.4f  |  Soft-NS=%.4f  |  F1=%.4f",
        total_softNC / n, total_softNS / n, total_f1 / n,
    )
    logger.info("=" * 60)

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, f"{args.dataset}_{args.model.replace('/', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(results_log, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HETA: Hessian-Enhanced Token Attribution — Evaluation Script"
    )
    p.add_argument("--model", type=str, default="gpt2",
                   help="HuggingFace model name or path.")
    p.add_argument("--dataset", type=str, default="tellmewhy",
                   choices=["longra", "tellmewhy", "wikibio", "curated"],
                   help="Evaluation dataset.")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Weight for Hessian sensitivity (Eq. 5).")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Weight for KL information (Eq. 5).")
    p.add_argument("--hvp-samples", type=int, default=10,
                   help="Hutchinson HVP samples (m).")
    p.add_argument("--mask-scheme", type=str, default="zero",
                   choices=["zero", "mean", "unk"])
    p.add_argument("--window-size", type=int, default=0,
                   help="Sliding window size (0 = disabled).")
    p.add_argument("--window-overlap", type=float, default=0.5)
    p.add_argument("--low-rank", type=int, default=0,
                   help="Low-rank Hessian approximation rank (0 = full).")
    p.add_argument("--layer-subset", type=int, default=0,
                   help="Compute curvature on last N layers only (0 = all).")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="results/",
                   help="Output directory for result logs.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
