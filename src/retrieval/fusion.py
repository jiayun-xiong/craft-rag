"""Reciprocal rank fusion of chunk IDs. Ranks are one-based."""
from collections import defaultdict
from math import fsum


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60) -> list[dict]:
    if k < 0:
        raise ValueError("RRF rank constant must be nonnegative")
    ranks = defaultdict(dict)
    for route, ids in rankings.items():
        for rank, cid in enumerate(dict.fromkeys(ids), 1):
            ranks[cid][route] = rank

    results = [
        {"id": cid, "rrf_score": fsum(1 / (k + r) for r in by_route.values()),
         "ranks": by_route}
        for cid, by_route in ranks.items()
    ]
    return sorted(results, key=lambda hit: (-hit["rrf_score"], hit["id"]))
