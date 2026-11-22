"""Parse text PDFs. Coordinates use PDF points from the top left of a page."""
from pathlib import Path
import re

import pdfplumber
import pymupdf


def extract_table(page, table):
    # Open-sided grids lose the outer column below the header. Restrict the
    # reconstructed edge to this table's own rectangle.
    box = table.bbox
    crop = page.crop((max(page.bbox[0],box[0]-2), max(page.bbox[1],box[1]-2),
                      min(page.bbox[2],box[2]+2), min(page.bbox[3],box[3]+2)))
    repaired = crop.find_tables({"explicit_vertical_lines": [box[0], box[2]]})
    for candidate in repaired:
        if (len(candidate.rows) == len(table.rows)
                and len(candidate.columns) == len(table.columns)
                and max(abs(a-b) for a, b in zip(candidate.bbox, box)) < 1):
            table = candidate
            break
    raw = table.extract()
    xs = [col.bbox[0] for col in table.columns]
    ys = [row.bbox[1] for row in table.rows]
    rows = [row.copy() for row in raw]
    # Propagate labels only inside a physically merged cell.
    for r, row in enumerate(table.rows):
        for c, rect in enumerate(row.cells):
            if rect and raw[r][c]:
                for i in range(r, len(ys)):
                    if ys[i] >= rect[3] - 0.1:
                        break
                    for j in range(c, len(xs)):
                        if xs[j] >= rect[2] - 0.1:
                            break
                        if raw[i][j] is None:
                            rows[i][j] = raw[r][c]
    above = page.crop((page.bbox[0], max(page.bbox[1], box[1]-40), page.bbox[2], box[1]))
    return {"bbox": box, "preceding_text": above.extract_text() or "",
            "raw_rows": raw, "rows": rows}


def link_continuations(pages):
    for prev, curr in zip(pages, pages[1:]):
        if curr["page"] != prev["page"] + 1 or not prev["tables"] or not curr["tables"]:
            continue
        a, b = prev["tables"][-1], curr["tables"][0]
        if (a["bbox"][3] < 0.75 * prev["height"] or b["bbox"][1] > 0.25 * curr["height"]
                or a["rows"][0] != b["rows"][0]):
            continue
        if re.search(r"续表|表.*?\(续\)|continued", b["preceding_text"], re.I):
            # Keep per-row page provenance when creating the logical table.
            chain = a.get("logical_id", a["id"])
            a["logical_id"] = b["logical_id"] = chain
        else:
            b["warning"] = "possible_continuation_review"


class PdfParser:
    def parse(self, path: Path, selected=None):
        pages = []
        with pymupdf.open(path) as doc, pdfplumber.open(path) as pdf:
            numbers = sorted(set(selected)) if selected is not None else range(1, len(doc)+1)
            for n in numbers:
                if not 1 <= n <= len(doc):
                    raise ValueError(f"页码 {n} 超出 1 至 {len(doc)}")
                page = doc[n-1]
                blocks = [{"bbox": list(b[:4]), "text": b[4].strip()}
                          for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]
                grid = pdf.pages[n-1]
                tables = []
                try:
                    for i, table in enumerate(grid.find_tables(), 1):
                        if len(table.rows) >= 2 and len(table.columns) >= 2:
                            tables.append({"id": f"p{n}-t{i}", **extract_table(grid, table)})
                finally:
                    grid.close()
                pages.append({"source": path.name, "page": n, "height": page.rect.height,
                              "width": page.rect.width, "blocks": blocks, "tables": tables,
                              "warnings": [] if blocks else ["no_text_layer_review"]})
        link_continuations(pages)
        return pages


def joined_tables(pages):
    groups = {}
    for page in pages:
        for t in page["tables"]:
            if "logical_id" in t:
                group = groups.setdefault(t["logical_id"], [])
                group.extend({"page": page["page"], "table": t["id"], "row": row}
                             for row in (t["rows"][1:] if group else t["rows"]))
    return groups
