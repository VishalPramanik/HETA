#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""
Quick Demo — HETA Attribution with GPT-2.

A self-contained demo that loads GPT-2 (124 M params), runs HETA on a few
example sentences, prints the top-k attributed tokens, and optionally writes
an HTML visualisation.

Usage:
    python scripts/demo.py
    python scripts/demo.py --model gpt2-medium --device cuda
    python scripts/demo.py --html output.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heta import HETA
from heta.utils import HETAConfig
from heta.visualization import print_attribution, to_html


EXAMPLES = [
    "Cam ordered a pizza and took it home. He opened the box to take "
    "out a slice. Cam discovered that the store did not cut the pizza "
    "for him. He had to use his chef knife to cut a slice",

    "Sandra got a job at the zoo. She loved coming to work and seeing "
    "all of the animals. She took pictures and shared them with her friends",

    "The capital of France is Paris",

    "I thought I lost my hat at the park today. I spent a lot of time "
    "looking for it. I was just about to give up when I saw something "
    "far away. It was my hat, stuck in a bush",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="HETA Quick Demo")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name.")
    parser.add_argument("--device", default="auto", help="Device (auto|cuda|cpu).")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--hvp-samples", type=int, default=5,
                        help="Hutchinson samples (fewer = faster demo).")
    parser.add_argument("--html", type=str, default=None,
                        help="Path to write HTML visualisation.")
    args = parser.parse_args()

    config = HETAConfig(
        beta=args.beta,
        gamma=args.gamma,
        num_hvp_samples=args.hvp_samples,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"  HETA Demo — Model: {args.model}")
    print(f"  β={config.beta}, γ={config.gamma}, HVP samples={config.num_hvp_samples}")
    print(f"{'='*60}\n")

    print("Loading model...")
    attributor = HETA.from_pretrained(args.model, config=config)
    print(f"Model loaded on {attributor.device}\n")

    for i, text in enumerate(EXAMPLES):
        print(f"\n--- Example {i + 1} ---")
        print(f"Input: {text}\n")

        result = attributor.attribute(text)

        # Console output
        print_attribution(
            result.tokens,
            result.scores,
            result.target_token,
            top_k=10,
        )

        # Component breakdown for top-5 tokens
        top_vals, top_idxs = result.scores.topk(min(5, len(result.scores)))
        print("  Component breakdown (top 5):")
        print(f"  {'Token':>15s}  {'M_T':>8s}  {'S_i':>8s}  {'I_i':>8s}  {'Final':>8s}")
        print(f"  {'-'*55}")
        for v, idx in zip(top_vals, top_idxs):
            j = idx.item()
            print(
                f"  {result.tokens[j]:>15s}"
                f"  {result.semantic_flow[j].item():8.4f}"
                f"  {result.hessian_sensitivity[j].item():8.4f}"
                f"  {result.kl_divergence[j].item():8.4f}"
                f"  {v.item():8.4f}"
            )
        print()

    # Optional HTML output
    if args.html:
        # Write the last example as HTML
        html = to_html(
            result.tokens, result.scores, result.target_token,
            output_path=args.html,
        )
        print(f"HTML visualisation saved to {args.html}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
