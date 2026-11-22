"""Export parsed pages to a UTF-8 JSON file."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.parsing.parser import PdfParser, joined_tables


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pdf", type=Path)
    cli.add_argument("--pages", nargs="+", type=int, help="PDF 页序，从 1 开始。默认全部页面")
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    if args.output.suffix.lower() != ".json":
        cli.error("输出文件须使用 .json 后缀")
    pages = PdfParser().parse(args.pdf, args.pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=args.output.parent, encoding="utf-8", delete=False) as f:
        json.dump({"pages": pages, "joined_tables": joined_tables(pages)}, f, ensure_ascii=False, indent=2)
    Path(f.name).replace(args.output)
    print(f"{args.output}：{len(pages)} 页，{sum(len(p['tables']) for p in pages)} 个候选表格")


if __name__ == "__main__":
    main()
