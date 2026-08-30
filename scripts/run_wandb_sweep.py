from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wandb

from ege_ah_mala.config import load_config
from ege_ah_mala.experiment import run_experiment


def _yaml_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="W&B 参数扫描运行包装器")
    parser.add_argument("--base-config", type=Path, required=True)
    args = parser.parse_args()
    base = load_config(args.base_config)
    run = wandb.init(
        project=base.wandb.project,
        entity=base.wandb.entity,
        group=base.wandb.group,
        tags=[*base.wandb.tags, "sweep"],
        job_type="sweep",
        mode="online",
    )
    try:
        parameters = dict(wandb.config)
        digest = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        overrides = [f"{key}={_yaml_value(value)}" for key, value in parameters.items()]
        overrides.extend(
            [
                "sampler.methods=[A4]",
                "wandb.mode=online",
                f"output.experiment_name=wandb_sweep_{digest}",
            ]
        )
        config = load_config(args.base_config, overrides)
        run.name = f"A4-sweep-{digest}"
        run_experiment(config)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
