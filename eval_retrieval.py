"""Development-set evidence-page recall; report row rank separately."""
import argparse
import json
from search import search
from src.retrieval.store import DEFAULT_INDEX, ROOT, save


def evaluate(m4=False, backend="api"):
    questions = ROOT/"data/eval/车削开发核对样例.jsonl"
    results = []
    for line in questions.read_text().splitlines():
        q = json.loads(line)
        expected = set(q.get("pdf_pages", [q.get("pdf_page")]))
        for route, hits in search(q["question"], DEFAULT_INDEX, "ablation" if m4 else "compare", 5, backend=backend).items():
            found = {h["page"] for h in hits if h["source"] == q["source"]}
            row_rank = None
            if q["id"] == "M1-01":
                row_rank = next((i for i,h in enumerate(hits,1)
                                 if h["source"] == q["source"] and h["page"] == 19
                                 and h.get("table") == "p19-t1" and h.get("row") == 2), None)
            result = dict(question=q["id"], route=route, all_pages_found=expected <= found,
                          target_row_rank=row_rank,
                          evidence_page_hits=sum(h["source"] == q["source"] and h["page"] in expected
                                                 for h in hits), hits=hits)
            print(q["id"], route, result["all_pages_found"], row_rank, flush=True)
            results.append(result)
    out = ROOT / ("output/M4/" + backend if m4 else "output/M3")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "development-results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4", action="store_true")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    args = parser.parse_args()
    evaluate(args.m4, args.backend)
