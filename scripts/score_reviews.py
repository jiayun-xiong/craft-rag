"""Summarize a completed human review sheet; never count blank ratings as passes."""
import argparse
import csv
import json
from pathlib import Path

def summarize(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or len({r["编号"] for r in rows})!=len(rows):
        raise ValueError("评分表为空或编号重复")
    cols=["参数/事实正确(0/1)","引用支持回答(0/1)"]
    done=[]
    for row in rows:
        scores=[row[col].strip() for col in cols]
        if not any(scores): continue
        if any(s not in ("0","1") for s in scores) or not row["评审人"].strip():
            raise ValueError("请补全两项评分及评审人："+row["编号"])
        done.append([int(s) for s in scores])
    return dict(总题数=len(rows),已复核=len(done),待复核=len(rows)-len(done),
                已评子集均分={col:sum(r[i] for r in done)/len(done) if done else None for i,col in enumerate(cols)})

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('sheet',type=Path)
    print(json.dumps(summarize(p.parse_args().sheet),ensure_ascii=False,indent=2))
