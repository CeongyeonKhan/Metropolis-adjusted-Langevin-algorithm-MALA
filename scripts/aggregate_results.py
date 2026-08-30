from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总多个种子和方法的结果")
    parser.add_argument("root", type=Path, help="包含 summary.json 的实验根目录")
    parser.add_argument("--output", type=Path, default=Path("aggregate_summary.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(path)
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"在 {args.root} 下未找到 summary.json")
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"已汇总 {len(frame)} 个方法运行：{args.output}")


if __name__ == "__main__":
    main()
