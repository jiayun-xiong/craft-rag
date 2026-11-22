"""M0 smoke checks. Never print credentials, response bodies, or exception text."""
import argparse
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Validate configuration only')
    args = parser.parse_args()
    config = {**dotenv_values(ROOT / '.env'), **os.environ}
    required = ['USTC_API_BASE', 'USTC_API_KEY', 'USTC_MODEL',
                'SILICONFLOW_API_BASE', 'SILICONFLOW_API_KEY', 'EMBED_MODEL', 'RERANK_MODEL']
    bad = [k for k in required if not config.get(k) or '在这里' in config[k]]
    for key in ['USTC_API_BASE', 'SILICONFLOW_API_BASE']:
        url = urlparse(config.get(key) or '')
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query:
            bad.append(key)
    if bad:
        print('配置缺失或无效: ' + ', '.join(sorted(set(bad))))
        return 1
    print('配置检查通过（不代表密钥鉴权成功）', flush=True)
    if args.offline:
        return 0

    checks = [
        ('embedding', 'SILICONFLOW', '/embeddings',
         {'model': config['EMBED_MODEL'], 'input': ['机械加工工艺资料检索测试。']}),
        ('rerank', 'SILICONFLOW', '/rerank',
         {'model': config['RERANK_MODEL'], 'query': '机械加工工艺',
          'documents': ['机械加工工序与刀具选择。', '今天的天气晴朗。'], 'top_n': 2}),
        ('generation', 'USTC', '/chat/completions',
         {'model': config['USTC_MODEL'], 'messages': [{'role': 'user', 'content': '请只回复 OK'}],
          'max_tokens': 128, 'stream': False}),
    ]
    failures = 0
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for name, provider, path, payload in checks:
            start = time.perf_counter()
            try:
                response = client.post(config[provider + '_API_BASE'].rstrip('/') + path,
                                       headers={'Authorization': 'Bearer ' + config[provider + '_API_KEY']},
                                       json=payload)
                response.raise_for_status()
                data = response.json()
                if name == 'embedding':
                    rows = data['data']
                    vector = rows[0]['embedding']
                    valid = len(rows) == 1 and len(vector) == 1024 and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector)
                    detail = f'dimensions={len(vector)}'
                elif name == 'rerank':
                    rows = data['results']
                    scores = [r['relevance_score'] for r in rows]
                    valid = (len(rows) == 2 and {r['index'] for r in rows} == {0, 1}
                             and all(isinstance(s, (float, int)) and math.isfinite(s) for s in scores)
                             and scores == sorted(scores, reverse=True) and rows[0]['index'] == 0)
                    detail = f'results={len(rows)}, expected_top={rows[0]["index"] == 0}'
                else:
                    content = data['choices'][0]['message']['content']
                    valid = isinstance(content, str) and bool(content.strip())
                    detail = f'nonempty={valid}, exact_OK={content.strip() == "OK" if valid else False}'
                if not valid:
                    raise ValueError('invalid response')
                print(f'{name}: PASS {time.perf_counter()-start:.2f}s {detail}', flush=True)
            except httpx.HTTPStatusError as exc:
                failures += 1
                print(f'{name}: FAIL HTTP {exc.response.status_code} {time.perf_counter()-start:.2f}s', flush=True)
            except Exception as exc:
                failures += 1
                print(f'{name}: FAIL {type(exc).__name__} {time.perf_counter()-start:.2f}s', flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
