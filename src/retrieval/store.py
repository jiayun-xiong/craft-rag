import json
import math
import os
import re
import unicodedata
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import dotenv_values
import httpx
import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "output/M2"
PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*|[\u3400-\u9fff]+", re.I)


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(\d)[～~─–—至](?=\d)", r"\1 ", text)
    tokens = []
    for word in PATTERN.findall(text):
        tokens.extend(jieba.lcut_for_search(word) if word[0] >= "\u3400" else [word])
    return tokens


def sparse_index(chunks):
    corpus = [tokenize(c["text"]) for c in chunks]
    model = BM25Okapi(corpus)
    postings = {}
    for i, freq in enumerate(model.doc_freqs):
        norm = model.k1 * (1 - model.b + model.b * model.doc_len[i] / model.avgdl)
        for term, count in freq.items():
            weight = model.idf[term] * count * (1 + model.k1) / (count + norm)
            postings.setdefault(term, []).append((chunks[i]["id"], weight))
    return postings


def sparse_search(postings, query, k):
    scores = {}
    for term in tokenize(query):
        for cid, value in postings.get(term, []):
            scores[cid] = scores.get(cid, 0) + value
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]


def config():
    return {**dotenv_values(ROOT / ".env"), **os.environ}


def vectors(client, cfg, texts):
    resp = client.post(cfg["SILICONFLOW_API_BASE"].rstrip("/") + "/embeddings",
                       headers={"Authorization": "Bearer " + cfg["SILICONFLOW_API_KEY"]},
                       json={"model": cfg["EMBED_MODEL"], "input": texts})
    resp.raise_for_status()
    rows = sorted(resp.json()["data"], key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(len(texts))):
        raise ValueError("嵌入返回顺序或条数错误")
    result = [r["embedding"] for r in rows]
    if any(len(v) != 1024 or not all(math.isfinite(x) for x in v) for v in result):
        raise ValueError("嵌入维度或数值无效")
    return result


def collection(path, create=False):
    db = chromadb.PersistentClient(path=str(path/"chroma"), settings=Settings(anonymized_telemetry=False))
    if create:
        return db.create_collection("craft_chunks", embedding_function=None,
                                    metadata={"hnsw:space":"cosine"})
    return db.get_collection("craft_chunks", embedding_function=None)


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf8")
