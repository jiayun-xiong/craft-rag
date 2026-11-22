"""Evidence units from M1. Table rows point back to the intact M1 table."""
import hashlib
import json
import re


def clean(value):
    return " ".join((value or "").split())


def windows(text, limit=500, overlap=75):
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            split = max(text.rfind(s, start + limit//2, end) for s in ("\n", "。", ". ", "；", " "))
            if split >= 0:
                end = split + 1
        yield text[start:end].strip()
        if end == len(text):
            break
        start = end - overlap


def encloses(outer, inner):
    return (inner[0] >= outer[0]-1 and inner[1] >= outer[1]-1 and
            inner[2] <= outer[2]+1 and inner[3] <= outer[3]+1)


def page_chunks(page):
    source, n = page["source"], page["page"]

    def chunk(kind, section, location, content, **extra):
        key = json.dumps([source, n, kind, location, content], ensure_ascii=False)
        return dict(id=hashlib.sha256(key.encode()).hexdigest(), source=source,
                    page=n, section=section, kind=kind, text=content, **extra)

    accepted = []
    for table in page["tables"]:
        rows = table["rows"]
        captions = [s.strip() for s in table["preceding_text"].splitlines()
                    if re.search(r"^Table\s+\d|刀片|切削条件|^\(mm\)$", s, re.I)]
        # Only promote captioned grids with a complete single header.
        # Complex header geometry and drawings remain in the original page text.
        if not captions or not all(clean(h) for h in table["raw_rows"][0] if h is not None):
            continue
        if not all(clean(h) for h in rows[0]) or (rows[0][0] == "No." and not clean(rows[1][0]).isdigit()):
            continue
        section = " ".join(captions)
        for i, row in enumerate(rows[1:], 2):
            pairs = list(dict.fromkeys(f"{clean(h)}: {clean(v)}" for h,v in zip(rows[0], row) if clean(v)))
            if pairs:
                yield chunk("table_row", section, [table["id"], i], section+"\n"+" | ".join(pairs),
                            table=table["id"], row=i)
        accepted.append(table["bbox"])
    body = "\n".join(b["text"] for b in page["blocks"]
                     if not any(encloses(box, b["bbox"]) for box in accepted))
    for i, part in enumerate(windows(body)):
        if part:
            yield chunk("text", f"PDF 第 {n} 页正文", i, part)
