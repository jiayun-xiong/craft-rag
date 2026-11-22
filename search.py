"""Retrieve and inspect dense, BM25 and fused results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import httpx
from src.retrieval.rerank import rerank
from src.retrieval.fusion import rrf_fuse
from src.retrieval.store import DEFAULT_INDEX, collection, config, load, sparse_search, vectors


def search(query, root, mode, k, candidate_k=20, backend="api"):
    if mode not in {"dense", "bm25", "both", "hybrid", "compare", "rerank", "ablation"}:
        raise ValueError("未知检索模式")
    if not query.strip() or min(k, candidate_k) < 1:
        raise ValueError("查询不能为空，返回数量必须为正")
    dest = root / (root / "CURRENT").read_text().strip()
    chunks = load(dest / "chunks.json")
    fusion = mode in {"hybrid", "compare", "rerank", "ablation"}
    depth = min(len(chunks), max(k, candidate_k) if fusion else k)

    def bm25():
        return sparse_search(load(dest / "bm25.json"), query, depth)

    def dense():
        cfg = config()
        if cfg["EMBED_MODEL"] != load(dest / "manifest.json")["model"]:
            raise ValueError("查询模型与索引模型不同")
        with httpx.Client(timeout=45) as client:
            vec = vectors(client, cfg, [query])
        result = collection(dest).query(query_embeddings=vec, n_results=depth, include=["distances"])
        return list(zip(result["ids"][0], result["distances"][0]))

    if mode in {"bm25", "dense"}:
        rankings = {mode: bm25() if mode == "bm25" else dense()}
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sparse_task = pool.submit(bm25)
            rankings = {"dense": dense(), "bm25": sparse_task.result()}

    output = {}
    if mode not in {"hybrid", "rerank", "ablation"}:
        for route, hits in rankings.items():
            metric = "distance" if route == "dense" else "score"
            output[route] = [{**chunks[cid], metric: value} for cid, value in hits[:k]]
    if fusion:
        fused = rrf_fuse({route: [cid for cid, _ in hits] for route, hits in rankings.items()})
        pool = [{**chunks[hit["id"]], **hit} for hit in fused[:depth]]
        if mode != "rerank":
            output["hybrid"] = pool[:k]
        if mode in {"rerank", "ablation"}:
            output["rerank"] = rerank(query, pool, k, backend)

    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--mode", choices=["dense", "bm25", "both", "hybrid", "compare", "rerank", "ablation"], default="both")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=20, help="融合前每路深度，至少为 k")
    p.add_argument("--backend", choices=["api", "local"], default="api")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    a = p.parse_args()
    if min(a.k, a.candidate_k) < 1 or not a.query.strip():
        p.error("输入非空问题及正数排名深度")
    print(json.dumps(search(a.query, a.index, a.mode, a.k, a.candidate_k, a.backend), ensure_ascii=False, indent=2))
