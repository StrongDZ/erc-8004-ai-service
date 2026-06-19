"""Unit tests for Stage 3: value_scale heuristic (no-metadata fallback).

Run with:
    cd erc-8004-ai-service
    .venv/bin/python3 -m unittest tests.test_stage3_heuristic -v
"""
from __future__ import annotations

import unittest

from benchmarks.stage3_domain import scale_heuristic


class TestScaleHeuristic(unittest.TestCase):
    def test_unbounded_is_quantity(self):
        self.assertEqual(scale_heuristic("unbounded", 0), "quantity")

    def test_positive_decimals_is_quantity(self):
        self.assertEqual(scale_heuristic("pct100", 2), "quantity")

    def test_star5_is_quality(self):
        self.assertEqual(scale_heuristic("star5", 0), "quality")

    def test_star10_is_quality(self):
        self.assertEqual(scale_heuristic("star10", 0), "quality")

    def test_binary_is_quality(self):
        self.assertEqual(scale_heuristic("binary", 0), "quality")

    def test_pct100_no_decimals_escalates(self):
        self.assertIsNone(scale_heuristic("pct100", 0))

    def test_empty_scale_escalates(self):
        self.assertIsNone(scale_heuristic("", 0))


if __name__ == "__main__":
    unittest.main()
