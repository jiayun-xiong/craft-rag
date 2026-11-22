import argparse
import json
from src.generation.pipeline import ask, answer_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = ask(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else answer_text(result))
    if not args.json:
        for i, ref in result["citations"].items():
            print(f'[{i}] {ref["source"]} · PDF 第 {ref["page"]} 页')
