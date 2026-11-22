"""Build both indexes into a new snapshot and publish only a complete pair."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import httpx
from src.parsing.parser import PdfParser
from src.retrieval.chunks import page_chunks
from src.retrieval.store import DEFAULT_INDEX, collection, config, load, save, sparse_index, vectors


def build(files, root):
    pages = {}
    for path in files:
        parsed = PdfParser().parse(path) if path.suffix.lower()==".pdf" else load(path)["pages"]
        for page in parsed:
            key = (page["source"], page["page"])
            if key in pages and pages[key] != page:
                raise ValueError(f"页来源冲突: {key}")
            pages[key] = page
    chunks = [c for key in sorted(pages) for c in page_chunks(pages[key])]
    if not chunks:
        raise ValueError("输入没有可索引的文字")
    cfg = config()
    root.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(("M2-v1"+cfg["EMBED_MODEL"]+"".join(c["id"] for c in chunks)).encode()).hexdigest()
    pointer = root/"CURRENT"
    if pointer.exists():
        current = root/pointer.read_text().strip()
        if load(current/"manifest.json")["signature"] == signature:
            return current
    dest = Path(tempfile.mkdtemp(prefix="build-", dir=root))
    coll = collection(dest, create=True)
    with httpx.Client(timeout=45) as client:
        for i in range(0, len(chunks), 16):
            batch = chunks[i:i+16]
            texts = [c["text"] for c in batch]
            coll.add(ids=[c["id"] for c in batch], documents=texts,
                     embeddings=vectors(client,cfg,texts),
                     metadatas=[{k:v for k,v in c.items() if k not in ("text","id")} for c in batch])
    postings = sparse_index(chunks)
    ids = {c["id"] for c in chunks}
    if ids != set(coll.get(include=[])["ids"]) or ids != {i for hits in postings.values() for i,_ in hits}:
        raise ValueError("索引 ID 不一致")
    save(dest/"chunks.json", {c["id"]:c for c in chunks})
    save(dest/"bm25.json", postings)
    save(dest/"manifest.json", dict(signature=signature, model=cfg["EMBED_MODEL"], chunks=len(chunks),
                                   pages=sorted(pages)))
    tmp = dest/"pointer"
    tmp.write_text(dest.name)
    os.replace(tmp, pointer)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path, help="M1 JSON 或完整文本型 PDF")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = p.parse_args()
    print(build(args.files, args.index))
