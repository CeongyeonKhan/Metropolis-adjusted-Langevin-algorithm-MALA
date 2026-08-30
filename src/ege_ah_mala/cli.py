from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ege-mala",
        description="能量—梯度误差约束自适应混合 MALA 实验",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="运行一个配置文件")
    run.add_argument("--config", required=True, type=Path, help="YAML 配置文件")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置，格式为 section.key=value；可重复给出",
    )
    run.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    run.add_argument("--device", help="例如 cpu、cuda 或 cuda:0")
    run.add_argument("--methods", help="逗号分隔的方法编号")
    run.add_argument("--steps", type=int, help="正式采样步数")
    run.add_argument("--chains", type=int, help="并行链数")
    run.add_argument("--output-root", help="输出根目录")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)
    if args.wandb_mode:
        overrides.append(f"wandb.mode={args.wandb_mode}")
    if args.device:
        overrides.append(f"device={args.device}")
    if args.methods:
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        overrides.append(f"sampler.methods={methods}")
    if args.steps is not None:
        overrides.append(f"sampler.num_steps={args.steps}")
    if args.chains is not None:
        overrides.append(f"sampler.num_chains={args.chains}")
    if args.output_root:
        overrides.append(f"output.root={args.output_root}")
    config = load_config(args.config, overrides)
    output = run_experiment(config)
    print(f"实验完成，结果目录：{output}")


if __name__ == "__main__":
    main()
