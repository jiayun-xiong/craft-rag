import unittest

from src.retrieval.fusion import rrf_fuse


class FusionTests(unittest.TestCase):
    def test_overlap_and_duplicate_in_one_route(self):
        hits = rrf_fuse({"dense": ["a", "a", "b"], "bm25": ["b", "c"]})
        self.assertEqual([h["id"] for h in hits], ["b", "a", "c"])
        self.assertEqual(hits[0]["ranks"], {"dense": 2, "bm25": 1})
        self.assertAlmostEqual(hits[0]["rrf_score"], 1/62 + 1/61)

    def test_ties_are_stable_and_empty_route_is_valid(self):
        ranks = {"dense": ["b", "a"], "bm25": ["a", "b"]}
        self.assertEqual(rrf_fuse(ranks), rrf_fuse(dict(reversed(list(ranks.items())))))
        self.assertEqual(rrf_fuse(ranks)[0]["id"], "a")
        self.assertEqual(rrf_fuse({"dense": [], "bm25": ["a"]})[0]["ranks"], {"bm25": 1})
        self.assertEqual(rrf_fuse({}), [])
