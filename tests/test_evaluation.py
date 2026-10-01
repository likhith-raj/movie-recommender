import unittest
import numpy as np
import pandas as pd
from scripts.evaluate import ranking_metrics, temporal_split

class EvaluationTests(unittest.TestCase):
    def test_perfect_ranking(self):
        recall, ndcg, precision = ranking_metrics([1, 2], {1, 2}, k=2)
        self.assertEqual((recall, ndcg, precision), (1., 1., 1.))

    def test_rank_discount(self):
        recall, ndcg, precision = ranking_metrics([9, 1], {1}, k=2)
        self.assertEqual(recall, 1.)
        self.assertAlmostEqual(ndcg, 1 / np.log2(3))
        self.assertEqual(precision, 0.5)

    def test_ndcg_agrees_with_sklearn(self):
        from sklearn.metrics import ndcg_score
        ranked = [3, 1, 2, 4]
        relevant = {1, 4}
        expected = ndcg_score([[1, 0, 0, 1]], [[3, 2, 4, 1]], k=3, ignore_ties=True)
        self.assertAlmostEqual(ranking_metrics(ranked, relevant, k=3)[1], expected)

    def test_missed_items_reduce_recall(self):
        self.assertEqual(ranking_metrics([1, 9], {1, 2}, k=2)[0], 0.5)

    def test_equal_timestamp_boundary_has_no_overlap(self):
        ratings = pd.DataFrame({"timestamp": [1, 2, 3, 3, 3, 4]})
        train, test, cutoff = temporal_split(ratings, fraction=0.5)
        self.assertLess(train.timestamp.max(), test.timestamp.min())
        self.assertEqual(len(train) + len(test), len(ratings))
        self.assertEqual(cutoff, 3)

    def test_duplicate_ranked_items_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([1, 1], {1})

if __name__ == "__main__":
    unittest.main()
