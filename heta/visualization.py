# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""Visualization utilities for HETA token-level attributions."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def print_attribution(
    tokens: List[str],
    scores: torch.Tensor,
    target_token: str,
    top_k: int = 15,
    show_bar: bool = True,
    bar_width: int = 30,
) -> None:
    """Pretty-print token attributions to the console.

    Args:
        tokens:       Context token strings.
        scores:       Per-token attribution scores (non-negative).
        target_token: The target token string.
        top_k:        Number of top tokens to display.
        show_bar:     Show a visual bar alongside scores.
        bar_width:    Character width of the visual bar.
    """
    max_score = scores.max().item() if scores.numel() > 0 else 1.0
    max_score = max(max_score, 1e-12)

    vals, idxs = scores.topk(min(top_k, len(scores)))

    print(f"\n{'='*60}")
    print(f"  Target token: {target_token!r}")
    print(f"  Top-{top_k} attributed context tokens:")
    print(f"{'='*60}")

    for rank, (v, i) in enumerate(zip(vals, idxs), start=1):
        tok = tokens[i.item()]
        score = v.item()
        norm = score / max_score

        if show_bar:
            filled = int(norm * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {rank:>3d}. {tok:>15s}  {score:8.4f}  {bar}")
        else:
            print(f"  {rank:>3d}. {tok:>15s}  {score:8.4f}")

    print(f"{'='*60}\n")


def to_html(
    tokens: List[str],
    scores: torch.Tensor,
    target_token: str,
    cmap: str = "Reds",
    output_path: Optional[str] = None,
) -> str:
    """Generate an HTML visualization of token attributions.

    Tokens are displayed with background colors proportional to their
    attribution scores, mirroring Figures 2(d), 3(d), 4–6 in the paper.

    Args:
        tokens:       Context token strings.
        scores:       Per-token scores.
        target_token: The target token.
        cmap:         Matplotlib colormap name.
        output_path:  If given, write the HTML to this file.

    Returns:
        HTML string.
    """
    max_score = scores.max().item() if scores.numel() > 0 else 1.0
    max_score = max(max_score, 1e-12)
    normed = (scores / max_score).clamp(0.0, 1.0).tolist()

    spans = []
    for tok, val in zip(tokens, normed):
        r = int(255 * val)
        g = int(255 * (1 - val * 0.7))
        b = int(255 * (1 - val * 0.7))
        bg = f"rgb({r},{g},{b})"
        spans.append(
            f'<span style="background-color:{bg};padding:2px 4px;'
            f'margin:1px;border-radius:3px;display:inline-block;'
            f'font-family:monospace;">{_escape(tok)}</span>'
        )

    # Target token in bold blue
    spans.append(
        f'<span style="background-color:#4a90d9;color:white;'
        f'padding:2px 6px;margin:1px;border-radius:3px;'
        f'display:inline-block;font-family:monospace;'
        f'font-weight:bold;">{_escape(target_token)}</span>'
    )

    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>HETA Attribution</title></head><body>"
        '<div style="max-width:800px;margin:40px auto;line-height:2.2;">'
        f'<h2 style="font-family:sans-serif;">HETA Attribution '
        f'→ Target: <em>{_escape(target_token)}</em></h2>'
        f'<p>{"".join(spans)}</p>'
        "</div></body></html>"
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML written to %s", output_path)

    return html


def _escape(text: str) -> str:
    """Basic HTML entity escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
