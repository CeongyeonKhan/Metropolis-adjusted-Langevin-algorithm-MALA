from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ege_ah_mala.calibration import generate_calibration_points
from ege_ah_mala.config import load_config
from ege_ah_mala.distributions import build_target
from ege_ah_mala.error_fields import GradientEnsemble
from ege_ah_mala.global_proposal import GlobalMixtureProposal
from ege_ah_mala.utils import make_generator, resolve_dtype


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(function, repeats: int, device: torch.device) -> float:
    for _ in range(3):
        function()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    _synchronize(device)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="测量等效梯度成本权重")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("cost_weights.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config.device)
    target = build_target(config.target, dtype=resolve_dtype(config.dtype), device=device)
    calibration = generate_calibration_points(
        target,
        config.calibration,
        make_generator(config.seed, "benchmark_calibration", device),
    )
    ensemble = GradientEnsemble.create(
        target,
        config.error,
        calibration,
        make_generator(config.seed, "benchmark_ensemble", device),
    )
    proposal = GlobalMixtureProposal(target, config.global_proposal)
    points = calibration[: args.points]
    vector = torch.ones_like(points) / max(1.0, target.dim**0.5)
    with torch.no_grad():
        energy_time = _measure(lambda: target.energy(points), args.repeats, device)
        gradient_time = _measure(lambda: ensemble.mean(points), args.repeats, device)
        hessian_time = _measure(
            lambda: ensemble.jacobian_vector_product_mean(points, vector),
            args.repeats,
            device,
        )
        global_time = _measure(lambda: proposal.log_prob(points), args.repeats, device)
    member_gradient_time = max(gradient_time / ensemble.ensemble_size, 1.0e-15)
    output = {
        "device": str(device),
        "dtype": config.dtype,
        "points_per_call": int(points.shape[0]),
        "seconds_per_batch": {
            "energy": energy_time,
            "gradient_ensemble": gradient_time,
            "gradient_hessian_vector_product": hessian_time,
            "global_density": global_time,
        },
        "recommended_cost_weights": {
            "energy_weight": energy_time / member_gradient_time,
            "hessian_weight": hessian_time / max(gradient_time, 1.0e-15),
            "global_density_weight": global_time / member_gradient_time,
        },
        "note": (
            "能量与全局密度权重以单个集成成员梯度为单位；HVP 权重利用同一集成大小"
            "的批次比值，成员数在计数器中显式展开。"
        ),
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"成本微基准已写入：{args.output}")


if __name__ == "__main__":
    main()
