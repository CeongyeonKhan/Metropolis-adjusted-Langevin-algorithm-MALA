from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any

from .config import WandbConfig
from .utils import json_ready


class ExperimentTracker:
    def __init__(
        self,
        run_dir: Path,
        config: WandbConfig,
        resolved_config: dict[str, Any],
        run_name: str,
        group: str,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = self.run_dir / "metrics.jsonl"
        self.local_handle = self.local_path.open("w", encoding="utf-8")
        self.wandb = None
        self.run = None
        self._owns_run = False
        if config.mode != "disabled":
            try:
                import wandb

                self.wandb = wandb
                wandb_dir = self.run_dir / "wandb"
                wandb_dir.mkdir(parents=True, exist_ok=True)
                if wandb.run is not None:
                    self.run = wandb.run
                    self.run.config.update(json_ready(resolved_config), allow_val_change=True)
                else:
                    self.run = wandb.init(
                        project=config.project,
                        entity=config.entity,
                        name=run_name,
                        group=config.group or group,
                        tags=config.tags,
                        job_type="sampling",
                        mode=config.mode,
                        dir=str(wandb_dir),
                        config=json_ready(resolved_config),
                    )
                    self._owns_run = True
                self._define_metrics()
            except Exception as exc:
                if config.strict:
                    raise
                message = f"W&B 初始化失败，已继续使用本地记录：{type(exc).__name__}: {exc}"
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                (self.run_dir / "wandb_error.txt").write_text(message, encoding="utf-8")
                self.wandb = None
                self.run = None

    def _define_metrics(self) -> None:
        assert self.wandb is not None
        self.wandb.define_metric("axis/ceq_per_chain")
        self.wandb.define_metric("axis/transition_per_chain")
        patterns = (
            "sampler/*",
            "state/*",
            "error/*",
            "accept/*",
            "constraints/*",
            "diagnostics/*",
            "distribution/*",
            "crossing/*",
            "cost/*",
            "runtime/*",
            "health/*",
        )
        for pattern in patterns:
            self.wandb.define_metric(pattern, step_metric="axis/ceq_per_chain")

    def log(self, metrics: dict[str, Any]) -> None:
        local_payload = json_ready(metrics)
        self.local_handle.write(
            json.dumps(local_payload, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.local_handle.flush()
        if self.run is None:
            return
        cloud_payload: dict[str, Any] = {}
        raw_rhat = metrics.get("diagnostics/rhat_max")
        if isinstance(raw_rhat, (int, float)) and math.isinf(float(raw_rhat)):
            cloud_payload["diagnostics/rhat_max_display_cap"] = 10.0
        for key, value in local_payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    cloud_payload[key] = value
            elif isinstance(value, (str, bool)):
                cloud_payload[key] = value
        self.run.log(cloud_payload)

    def log_images(self, image_paths: dict[str, Path]) -> None:
        if self.run is None or self.wandb is None:
            return
        payload = {
            key: self.wandb.Image(str(path)) for key, path in image_paths.items() if path.exists()
        }
        if payload:
            self.run.log(payload)

    def set_summary(self, summary: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in json_ready(summary).items():
            if (
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                or isinstance(value, (str, bool))
            ):
                self.run.summary[key] = value

    def log_artifact(self, files: list[Path], name: str) -> None:
        if self.run is None or self.wandb is None or not self.config.log_artifact:
            return
        artifact = self.wandb.Artifact(name=name, type="experiment-results")
        for path in files:
            if path.exists() and path.is_file():
                try:
                    artifact_name = str(path.relative_to(self.run_dir))
                except ValueError:
                    artifact_name = f"shared/{path.name}"
                artifact.add_file(str(path), name=artifact_name)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if not self.local_handle.closed:
            self.local_handle.close()
        if self.run is not None and self._owns_run:
            self.run.finish()

    def __enter__(self) -> ExperimentTracker:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish()
