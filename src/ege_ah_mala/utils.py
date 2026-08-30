from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def named_seed(base_seed: int, name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def make_generator(
    base_seed: int, name: str, device: torch.device | str = "cpu"
) -> torch.Generator:
    resolved_device = torch.device(device)
    generator_device: torch.device | str = (
        resolved_device if resolved_device.type == "cuda" else "cpu"
    )
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(named_seed(base_seed, name))
    return generator


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_ready(value.detach().cpu().item())
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for distribution in ("numpy", "scipy", "PyYAML", "matplotlib", "pandas", "wandb"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "packages": packages,
    }
