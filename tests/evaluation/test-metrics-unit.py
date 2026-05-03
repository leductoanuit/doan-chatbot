"""Unit tests for retrieval ranking metrics (retrieval-metrics.py).

All reference values hand-computed from standard IR textbook definitions.
Covers: precision@k, recall@k, hit@k, MRR, average_precision@k, MAP@k.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# retrieval-metrics.py uses a hyphen — import via importlib
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "retrieval_metrics", Path(__file__).parent / "retrieval-metrics.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

precision_at_k = _mod.precision_at_k
recall_at_k = _mod.recall_at_k
hit_at_k = _mod.hit_at_k
mrr = _mod.mrr
average_precision_at_k = _mod.average_precision_at_k
map_at_k = _mod.map_at_k


class TestPrecisionAtK(unittest.TestCase):
    def test_all_relevant(self):
        self.assertAlmostEqual(precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3), 1.0)

    def test_no_relevant(self):
        self.assertAlmostEqual(precision_at_k(["x", "y", "z"], {"a", "b"}, 3), 0.0)

    def test_half_relevant(self):
        # 2 hits in top-4 → 0.5
        self.assertAlmostEqual(precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4), 0.5)

    def test_k_larger_than_list(self):
        # Only 2 items, k=5 → 1 hit / 5
        self.assertAlmostEqual(precision_at_k(["a", "x"], {"a"}, 5), 0.2)

    def test_empty_retrieved(self):
        self.assertAlmostEqual(precision_at_k([], {"a"}, 3), 0.0)

    def test_k_zero(self):
        self.assertAlmostEqual(precision_at_k(["a", "b"], {"a"}, 0), 0.0)


class TestRecallAtK(unittest.TestCase):
    def test_all_found(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "b"}, 2), 1.0)

    def test_none_found(self):
        self.assertAlmostEqual(recall_at_k(["x", "y"], {"a", "b"}, 2), 0.0)

    def test_partial_recall(self):
        # 1 of 2 relevant found → 0.5
        self.assertAlmostEqual(recall_at_k(["a", "x", "y"], {"a", "b"}, 3), 0.5)

    def test_empty_retrieved(self):
        self.assertAlmostEqual(recall_at_k([], {"a"}, 3), 0.0)

    def test_empty_relevant(self):
        self.assertAlmostEqual(recall_at_k(["a", "b"], set(), 2), 0.0)

    def test_k_larger_than_list(self):
        # retrieved=["a"], relevant={"a","b"}, k=10 → 1/2
        self.assertAlmostEqual(recall_at_k(["a"], {"a", "b"}, 10), 0.5)


class TestHitAtK(unittest.TestCase):
    def test_first_is_hit(self):
        self.assertAlmostEqual(hit_at_k(["a", "b", "c"], {"a"}, 3), 1.0)

    def test_last_in_k_is_hit(self):
        self.assertAlmostEqual(hit_at_k(["x", "y", "a"], {"a"}, 3), 1.0)

    def test_no_hit(self):
        self.assertAlmostEqual(hit_at_k(["x", "y", "z"], {"a"}, 3), 0.0)

    def test_hit_outside_k(self):
        # "a" is at position 4, k=3 → miss
        self.assertAlmostEqual(hit_at_k(["x", "y", "z", "a"], {"a"}, 3), 0.0)

    def test_empty_retrieved(self):
        self.assertAlmostEqual(hit_at_k([], {"a"}, 3), 0.0)

    def test_empty_relevant(self):
        self.assertAlmostEqual(hit_at_k(["a"], set(), 3), 0.0)


class TestMRR(unittest.TestCase):
    def test_first_rank(self):
        # First hit at rank 1 → 1.0
        self.assertAlmostEqual(mrr(["a", "b", "c"], {"a"}), 1.0)

    def test_third_rank(self):
        # First hit at rank 3 → 1/3 ≈ 0.333
        self.assertAlmostEqual(mrr(["x", "y", "a"], {"a"}), 1 / 3, places=4)

    def test_no_hit(self):
        self.assertAlmostEqual(mrr(["x", "y", "z"], {"a"}), 0.0)

    def test_empty_retrieved(self):
        self.assertAlmostEqual(mrr([], {"a"}), 0.0)

    def test_empty_relevant(self):
        self.assertAlmostEqual(mrr(["a", "b"], set()), 0.0)

    def test_multiple_relevant_returns_first(self):
        # hits at 2 and 4 → MRR = 1/2
        self.assertAlmostEqual(mrr(["x", "a", "y", "b"], {"a", "b"}), 0.5)


class TestAveragePrecisionAtK(unittest.TestCase):
    def test_textbook_example(self):
        # retrieved=[a,b,c,d], relevant={a,c}, k=4
        # hits at pos 1,3 → P@1=1/1, P@3=2/3 → AP = (1 + 2/3) / min(2,4) = 0.8333
        result = average_precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4)
        self.assertAlmostEqual(result, (1.0 + 2 / 3) / 2, places=4)

    def test_all_relevant(self):
        self.assertAlmostEqual(average_precision_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_no_relevant_in_retrieved(self):
        self.assertAlmostEqual(average_precision_at_k(["x", "y"], {"a"}, 2), 0.0)

    def test_empty_retrieved(self):
        self.assertAlmostEqual(average_precision_at_k([], {"a"}, 3), 0.0)

    def test_empty_relevant(self):
        self.assertAlmostEqual(average_precision_at_k(["a", "b"], set(), 2), 0.0)

    def test_k_smaller_than_list(self):
        # Only top-2 of [a,b,c], relevant={a,c}: hit only "a" at pos 1 → AP = (1/1)/min(2,2) = 0.5
        result = average_precision_at_k(["a", "b", "c"], {"a", "c"}, 2)
        self.assertAlmostEqual(result, 0.5)


class TestMAP(unittest.TestCase):
    def test_single_query_perfect(self):
        self.assertAlmostEqual(map_at_k([["a", "b"]], [{"a", "b"}], 2), 1.0)

    def test_two_queries_average(self):
        # Q1: AP=1.0, Q2: AP=0.0 → MAP=0.5
        result = map_at_k(
            [["a", "b"], ["x", "y"]],
            [{"a"}, {"a"}],
            2,
        )
        self.assertAlmostEqual(result, 0.5)

    def test_empty_query_list(self):
        self.assertAlmostEqual(map_at_k([], [], 3), 0.0)

    def test_textbook_map(self):
        # Two queries, both perfect → 1.0
        result = map_at_k(
            [["a", "b", "c"], ["d", "e"]],
            [{"a", "b", "c"}, {"d", "e"}],
            3,
        )
        self.assertAlmostEqual(result, 1.0)

    def test_map_no_hits(self):
        result = map_at_k([["x", "y"], ["p", "q"]], [{"a"}, {"b"}], 2)
        self.assertAlmostEqual(result, 0.0)

    def test_map_k_larger_than_retrieved(self):
        # k=10 but only 2 items — should not error
        result = map_at_k([["a", "b"]], [{"a"}], 10)
        self.assertAlmostEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
