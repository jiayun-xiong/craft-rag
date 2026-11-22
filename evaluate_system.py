"""Run the end-to-end acceptance set; leave human scoring columns blank."""
import csv
import json
import math
from pathlib import Path
import statistics
import threading
import time

import psutil
from src.generation.pipeline import ask, answer_text
from src.retrieval.store import ROOT


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in (ROOT/'data/eval/端到端验收.jsonl').read_text().splitlines()]
    proc = psutil.Process()
    samples = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            samples.append(proc.memory_info().rss)
            stop.wait(0.1)
    monitor = threading.Thread(target=sample,daemon=True)
    monitor.start()
    results = []
    try:
        with (out/'answers.jsonl').open('w') as log, (out/'人工评分.csv').open('w',encoding='utf-8-sig',newline='') as sheet:
            writer=csv.writer(sheet)
            writer.writerow(['编号','问题','参考要点','回答','真实引用','参数/事实正确(0/1)','引用支持回答(0/1)','评审人','说明'])
            for row in rows:
                before=time.perf_counter()
                try:
                    answer=ask(row['question'])
                except Exception as exc:
                    answer={'status':'error','error':type(exc).__name__,'citations':{},'findings':[],'missing':[]}
                expected=set(row.get('pdf_pages',[row.get('pdf_page')]))
                found={r['page'] for r in answer['citations'].values() if r['source']==row['source']}
                record={'id':row['id'],'s':round(time.perf_counter()-before,3),'page_hit':expected<=found if row['source'] else None,'answer':answer}
                results.append(record)
                log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                writer.writerow([row['id'],row['question'],row['expected'],answer_text(answer) if answer['status']!='error' else '服务错误',
                                 '；'.join(f"{c['source']} p.{c['page']}" for c in answer['citations'].values()),'','','',''])
                sheet.flush()
                print(record['id'],answer['status'],record['s'],record['page_hit'],flush=True)
    finally:
        stop.set();monitor.join()
    times = [r['s'] for r in results if r['answer']['status'] != 'error']
    stats=dict(n=len(results),median_s=statistics.median(times) if times else None,
               p95_s=sorted(times)[math.ceil(.95*len(times))-1] if times else None,
               max_s=max(times, default=None),within15=sum(t<=15 for t in times),rss_peak_mib=max(samples)/2**20,
               errors=sum(r['answer']['status']=='error' for r in results),
               all_expected_pages=sum(r['page_hit'] is True for r in results),
               out_of_scope_refusals=sum(r['page_hit'] is None and r['answer']['status']=='insufficient' for r in results))
    (out/'metrics.json').write_text(json.dumps(stats,indent=2))
    print(stats)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path, required=True)
    run(p.parse_args().out)
