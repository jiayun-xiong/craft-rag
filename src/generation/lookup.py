"""Extractive answers for unambiguous, narrowly defined evidence lookups."""
import re
import unicodedata
from pathlib import Path
import pymupdf


def lookup(query, hits):
    ids = re.findall(r"\bO\d{3,6}\b", unicodedata.normalize("NFKC",query),re.I|re.ASCII)
    if len(ids)==1 and "程序号" in query and any(w in query for w in ("唯一","重名","重复")):
        refs={}
        for i,h in enumerate(hits,1):
            if re.search(r"(?<![A-Z0-9])"+ids[0]+r"(?![A-Z0-9])",h["text"],re.I):
                refs.setdefault((h["source"],h["page"]),i)
        if len(refs)>=2:
            return {"status":"answered","findings":[{"field":"程序号定位","value":f"{ids[0]} 出现在此处，仅凭编号不能唯一定位页码。","ref":i} for i in refs.values()],"missing":[]}

    # A grade with two distinct operating rows cannot determine one feed.
    if ("仅" in query or "只" in query) and "进给" in query and any(w in query for w in ("唯一","能否","确定")):
        grades = re.findall(r"(?<![A-Za-z])[A-Za-z]{2,}\d{2,}(?![A-Za-z0-9])", query)
        if len(grades)==1:
            ranges = {}
            for i,h in enumerate(hits,1):
                if "刀片材料: "+grades[0].upper()+" |" not in h["text"]: continue
                m=re.search(r"每转进给量 f \(mm/rev\): ([^|]+)",h["text"])
                if m: ranges.setdefault(m[1].strip(),i)
            if len(ranges)>1:
                return {"status":"clarify","findings":[{"field":"不同工况的参考行","value":v+" mm/rev","ref":i}
                        for v,i in list(ranges.items())[:2]],"missing":["仅凭牌号无法唯一选择进给。请提供材料、硬度、刀片类型、断屑槽及加工工况。"]}
    return None


def legend_context(query, hits, directory):
    """Read left-hand single-character labels on the same baseline as PDF text."""
    if not any(w in query for w in ("标注","图例","标识")): return hits
    output=[]
    for h in hits:
        path=directory/Path(h["source"]).name
        if h["kind"]!="text" or not path.is_file():
            output.append(h); continue
        with pymupdf.open(path) as doc:
            words=doc[h["page"]-1].get_text("words")
        labels=[]
        for x,y,right,_,w,*_ in words:
            if not re.fullmatch("[A-Z]",w): continue
            nearby=sorted((p for p in words if 0<p[0]-right<90 and abs(p[1]-y)<2.5),key=lambda p:p[0])
            sequence=[]
            for p in nearby:
                if p[0]-right>18 or not re.fullmatch("[A-Za-z]+",p[4]): break
                sequence.append(p[4]);right=p[2]
            phrase=" ".join(sequence)
            if sequence and phrase in h["text"]: labels.append(w+" = "+phrase)
        output.append({**h,"text":h["text"]+"\n同一图例行（按原页位置）："+" | ".join(labels)} if labels else h)
    return output
