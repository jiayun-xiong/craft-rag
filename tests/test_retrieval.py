import unittest
from rank_bm25 import BM25Okapi
from src.retrieval.store import tokenize, sparse_index, sparse_search
from src.retrieval.chunks import windows


class IndexTests(unittest.TestCase):
    def test_saved_postings_match_reference(self):
        texts = ["MC6015 FP 0.08 mm/rev", "CK3665 转序", "XD-40A 铣床", "MC6025 一般切削"]
        chunks = [dict(id=str(i), text=s) for i,s in enumerate(texts)]
        query = "XD-40A XD-40A MC6015"
        ref = BM25Okapi([tokenize(t) for t in texts]).get_scores(tokenize(query))
        ranked = dict(sparse_search(sparse_index(chunks), query, 4))
        for i, score in enumerate(ref):
            self.assertAlmostEqual(ranked.get(str(i), 0), score)
        self.assertIn("xd-40a",tokenize(query))

    def test_long_block_keeps_overlap_and_tail(self):
        parts = list(windows("甲"*1100+"末尾"))
        self.assertTrue(all(len(p)<=500 for p in parts))
        self.assertEqual(parts[-1][-2:], "末尾")
        self.assertEqual(sum(map(len,parts))-75*(len(parts)-1),1102)
