"""Cross-encoder scoring of an ordered RRF candidate pool."""
from functools import lru_cache
import math

import httpx
from .store import config


def ranked_hits(pool, rows, top_n):
    if len(rows) != len(pool):
        raise ValueError("重排响应条数与候选数不符")
    seen = set()
    hits = []
    for row in rows:
        idx, score = row["index"], row["relevance_score"]
        if type(idx) is not int or idx not in range(len(pool)) or idx in seen:
            raise ValueError("重排响应索引重复或越界")
        if type(score) not in (float, int) or not math.isfinite(score):
            raise ValueError("重排分数无效")
        seen.add(idx)
        hits.append({**pool[idx], "rrf_rank":idx+1, "rerank_score":score})
    return sorted(hits, key=lambda h: (-h["rerank_score"], h["rrf_rank"]))[:top_n]


@lru_cache(maxsize=1)
def local_model(name):
    from FlagEmbedding import FlagReranker
    return FlagReranker(name, use_fp16=False, devices=["cpu"])


def rerank(query, pool, top_n, backend="api"):
    if top_n < 1:
        raise ValueError("top_n 必须为正数")
    if not pool:
        return []
    cfg = config()
    model = cfg["RERANK_MODEL"]
    docs = [f"{h['source']} | PDF {h['page']}\n{h['text']}" for h in pool]
    if backend == "api":
        with httpx.Client(timeout=30) as client:
            res = client.post(cfg["SILICONFLOW_API_BASE"].rstrip("/") + "/rerank",
                              headers={"Authorization": "Bearer " + cfg["SILICONFLOW_API_KEY"]},
                              json={"model":model, "query":query, "documents":docs, "top_n":len(docs)})
            res.raise_for_status()
            rows = res.json()["results"]
    elif backend == "local":
        scores = local_model(model).compute_score([[query, doc] for doc in docs], normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        rows = [{"index":i, "relevance_score":float(s)} for i,s in enumerate(scores)]
    else:
        raise ValueError("未知重排后端")
    return ranked_hits(pool, rows, top_n)
