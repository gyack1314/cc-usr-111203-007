"""命令行：inspect保留原始摘要，modes输出面积加权低秩模态与留出结果。"""
import argparse
import json
from .io import load
from .domain import summarize
from .lowrank import analyze


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p_inspect = sub.add_parser('inspect')
    p_inspect.add_argument('file')
    p_modes = sub.add_parser('modes')
    p_modes.add_argument('file')
    a = parser.parse_args()
    v, digest = load(a.file)
    if a.command == 'inspect':
        print(json.dumps({'source_sha256': digest, 'summary': summarize(v)},
                         ensure_ascii=False, indent=2, allow_nan=False))
    else:
        result = analyze(v)
        result = {'source_sha256': digest, **result}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
