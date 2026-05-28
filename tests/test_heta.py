#!/usr/bin/env python3
# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# Licensed under the Apache License, Version 2.0

"""Unit tests for the HETA attribution framework."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heta.utils import (
    AttributionResult,
    HETAConfig,
    build_windows,
    mask_token_embedding,
    resolve_device,
)
from heta.metrics import DSA, f1_alignment


class TestHETAConfig(unittest.TestCase):
    """Validate configuration constraints."""

    def test_default_config(self):
        cfg = HETAConfig()
        self.assertEqual(cfg.beta, 0.5)
        self.assertEqual(cfg.gamma, 0.5)
        self.assertEqual(cfg.mask_scheme, "zero")

    def test_invalid_beta(self):
        with self.assertRaises(AssertionError):
            HETAConfig(beta=-1.0)

    def test_invalid_mask_scheme(self):
        with self.assertRaises(AssertionError):
            HETAConfig(mask_scheme="invalid")


class TestUtils(unittest.TestCase):
    """Test shared utility functions."""

    def test_resolve_device(self):
        dev = resolve_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_build_windows_no_overlap(self):
        windows = build_windows(100, 50, overlap=0.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0], (0, 50))
        self.assertEqual(windows[1], (50, 100))

    def test_build_windows_with_overlap(self):
        windows = build_windows(100, 60, overlap=0.5)
        # stride = 30, windows: [0,60], [30,90], [60,100]
        self.assertTrue(len(windows) >= 2)
        # First window
        self.assertEqual(windows[0], (0, 60))

    def test_build_windows_short_seq(self):
        windows = build_windows(20, 50, overlap=0.5)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], (0, 20))

    def test_mask_token_zero(self):
        emb = torch.randn(1, 5, 8)
        masked = mask_token_embedding(emb, token_idx=2, scheme="zero")
        self.assertTrue(torch.all(masked[0, 2, :] == 0))
        # Other tokens should be unchanged
        self.assertTrue(torch.allclose(masked[0, 0, :], emb[0, 0, :]))

    def test_mask_token_mean(self):
        emb = torch.randn(1, 5, 8)
        masked = mask_token_embedding(emb, token_idx=1, scheme="mean")
        expected = emb[0].mean(dim=0)
        self.assertTrue(torch.allclose(masked[0, 1, :], expected))


class TestAttributionResult(unittest.TestCase):
    """Test the result container."""

    def test_topk(self):
        scores = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])
        result = AttributionResult(
            scores=scores,
            semantic_flow=torch.zeros(5),
            hessian_sensitivity=torch.zeros(5),
            kl_divergence=torch.zeros(5),
            tokens=["a", "b", "c", "d", "e"],
            target_token="f",
            target_position=5,
        )
        top3 = result.topk(3)
        self.assertEqual(len(top3), 3)
        self.assertEqual(top3[0][0], "d")  # highest score = 0.8


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def test_dsa(self):
        scores = torch.tensor([0.1, 0.1, 0.3, 0.4, 0.1])
        relevant = [2, 3]     # mass = 0.3 + 0.4 = 0.7
        distractor = [0, 1]   # mass = 0.1 + 0.1 = 0.2
        dsa = DSA.evaluate(scores, relevant, distractor)
        # DSA = 0.7 - 0.2 = 0.5
        self.assertAlmostEqual(dsa, 0.5, places=4)

    def test_dsa_zero_scores(self):
        scores = torch.zeros(5)
        dsa = DSA.evaluate(scores, [0, 1], [3, 4])
        self.assertEqual(dsa, 0.0)

    def test_f1_perfect(self):
        scores = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
        gold = [2, 3]
        f1 = f1_alignment(scores, gold, top_k=2)
        self.assertAlmostEqual(f1, 1.0, places=4)

    def test_f1_no_overlap(self):
        scores = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        gold = [2, 3]
        f1 = f1_alignment(scores, gold, top_k=2)
        self.assertAlmostEqual(f1, 0.0, places=4)

    def test_f1_partial(self):
        scores = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
        gold = [1, 3]
        f1 = f1_alignment(scores, gold, top_k=2)
        # predicted = {1, 2}, gold = {1, 3}, tp=1, prec=0.5, rec=0.5
        self.assertAlmostEqual(f1, 0.5, places=4)


class TestIntegration(unittest.TestCase):
    """Integration test with GPT-2 (requires model download)."""

    @unittest.skipUnless(
        torch.cuda.is_available() or True,  # Always run; GPT-2 is small
        "Skipping integration test — requires transformers + model download.",
    )
    def test_gpt2_attribution(self):
        """Run a full HETA attribution pass on GPT-2."""
        try:
            from heta import HETA
            from heta.utils import HETAConfig

            cfg = HETAConfig(
                beta=0.5, gamma=0.5,
                num_hvp_samples=2,  # Minimal for speed
                device="cpu",
            )
            attributor = HETA.from_pretrained("gpt2", config=cfg)
            result = attributor.attribute("The capital of France is Paris")

            # Basic sanity checks
            self.assertTrue(result.scores.numel() > 0)
            self.assertTrue(result.scores.min() >= 0.0)
            self.assertEqual(len(result.tokens), len(result.scores))
            self.assertTrue(len(result.target_token) > 0)
            self.assertTrue(
                torch.isfinite(result.scores).all(),
                "Attribution scores contain NaN/Inf.",
            )
        except ImportError:
            self.skipTest("transformers not installed.")


if __name__ == "__main__":
    unittest.main()
