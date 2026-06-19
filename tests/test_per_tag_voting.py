"""Unit tests for the Stage 2 voting combiner (benchmarks.per_tag_svm.vote_per_tag).

Run with:
    cd erc-8004-ai-service
    .venv/bin/python3 -m unittest tests.test_per_tag_voting -v
"""
from __future__ import annotations

import unittest

from benchmarks.per_tag_svm import vote_per_tag


class TestVotePerTag(unittest.TestCase):
    def test_both_agree_quality(self):
        self.assertEqual(vote_per_tag(0.85, 0.80), "quality")

    def test_both_agree_non_quality(self):
        # Both confidently non_quality (p <= 1-thresh = 0.30) and they agree.
        self.assertEqual(vote_per_tag(0.05, 0.10), "non_quality")

    def test_both_in_uncertain_band_escalates(self):
        # Neither side reaches quality (>=0.70) nor non_quality (<=0.30).
        self.assertIsNone(vote_per_tag(0.60, 0.65))

    def test_real_conflict_escalates(self):
        # tag1 confidently quality (0.85 >= 0.70), tag2 confidently non_quality
        # (0.10 <= 0.30) — both confident, opposite classes → escalate to Stage 3.
        self.assertIsNone(vote_per_tag(0.85, 0.10))

    def test_only_tag1_confident(self):
        self.assertEqual(vote_per_tag(0.80, 0.50), "quality")

    def test_only_tag2_confident(self):
        self.assertEqual(vote_per_tag(0.40, 0.82), "quality")

    def test_empty_tag2_confident_tag1(self):
        self.assertEqual(vote_per_tag(0.80, 0.0, t2_empty=True), "quality")

    def test_empty_tag2_unconfident_tag1_escalates(self):
        self.assertIsNone(vote_per_tag(0.60, 0.0, t2_empty=True))


if __name__ == "__main__":
    unittest.main()
