#!/usr/bin/env python3
# [项目注释] 模块：命令行/流水线入口，负责执行 eval/select_checkpoint 阶段。
# [项目注释] 该文件的公共边界、输入输出和调用关系由下方实现及架构文档共同定义。

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq
from travel_grpo.evaluation.artifacts import atomic_json
from travel_grpo.evaluation.checkpoint_selection import select_checkpoint
from travel_grpo.evaluation.validation import summarize_validation_file

# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    select_checkpoint。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT/"data/grpo/validation.parquet")
    parser.add_argument("--sft-summary", type=Path)
    parser.add_argument("--candidate", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    if args.validation_dir:
        if args.sft_summary or args.candidate:
            parser.error("--validation-dir cannot be combined with manual summaries")
        raw = {
            int(path.stem): path
            for path in args.validation_dir.glob("*.jsonl")
            if path.stem.isdigit()
        }
        if 0 not in raw or not any(step > 0 for step in raw):
            raise ValueError("validation directory requires 0.jsonl and candidate step dumps")
        tasks = pq.read_table(args.tasks).to_pylist()
        summary_dir = args.validation_dir.parent / "validation_summaries"
        summaries = {}
        for step, path in sorted(raw.items()):
            summary = summarize_validation_file(path, tasks)
            destination = summary_dir / f"step_{step}.summary.json"
            atomic_json(destination, summary)
            summaries[step] = (destination, summary)
        args.sft_summary = summaries[0][0]
        args.candidate = [
            summaries[step][0] for step in sorted(summaries) if step > 0
        ]
        if args.output is None:
            args.output = args.validation_dir.parent / "checkpoint_selection.json"
    elif not args.sft_summary or not args.candidate:
        parser.error("provide --validation-dir or both --sft-summary and --candidate")
    if args.output is None:
        args.output = ROOT/"outputs/models/grpo/checkpoint_selection.json"
    candidates=[]
    for path in args.candidate:
        match=re.search(r"(?:step_|global_step_)(\d+)", str(path))
        if not match: raise ValueError(f"cannot infer checkpoint step from {path}")
        candidates.append({"step":int(match.group(1)), "summary_path":str(path), "summary":json.loads(path.read_text(encoding="utf-8"))})
    result=select_checkpoint(candidates, json.loads(args.sft_summary.read_text(encoding="utf-8")))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1
if __name__ == "__main__": raise SystemExit(main())
