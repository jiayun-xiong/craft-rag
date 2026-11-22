import argparse
from pathlib import Path
import json
from tempfile import mkdtemp
from src.nc.turning import Setup, compile_job


def export(dxf, setup, parent):
    nc, report = compile_job(dxf, setup)
    parent.mkdir(exist_ok=True, parents=True)
    dest = Path(mkdtemp(prefix='job-',dir=parent))
    (dest/'O0701.nc').write_text(nc, encoding='ascii')
    (dest/'工序卡.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return dest


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('dxf',type=Path)
    p.add_argument('setup',type=Path)
    p.add_argument('--out',type=Path,default=Path('output/jobs'))
    a=p.parse_args()
    print(export(a.dxf,Setup.model_validate_json(a.setup.read_text()),a.out))
