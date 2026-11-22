import unittest
from src.retrieval.rerank import ranked_hits


class RerankTests(unittest.TestCase):
    def test_out_of_order_results_retain_original_citation(self):
        pool = [{"page": 19}, {"page": 220}, {"page": 231}]
        rows = [{"index": i, "relevance_score": s} for i,s in [(2,.5),(1,.9),(0,.5)]]
        hits = ranked_hits(pool, rows, 2)
        self.assertEqual([(h["page"],h["rrf_rank"]) for h in hits], [(220,2),(19,1)])

    def test_invalid_mapping_rejected(self):
        row = {"index":0,"relevance_score":0.6}
        for rows in ([row,row], [row], [row,dict(row,index=2)], [row,dict(row,index=1,relevance_score=float("nan"))]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ranked_hits([{},{}], rows, 2)
