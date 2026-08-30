from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from ege_ah_mala.config import load_config
from ege_ah_mala.experiment import run_experiment


def test_end_to_end_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "smoke.yaml",
        [
            "sampler.methods=[B1,A4]",
            "sampler.num_chains=4",
            "sampler.num_steps=20",
            "sampler.checkpoints=[0,5,10,20]",
            "sampler.diagnostic_stride=1",
            "calibration.num_points=80",
            "error.features=8",
            "metrics.fid_reference_repeats=2",
            "wandb.mode=disabled",
            "wandb.log_plots=false",
            f"output.root={tmp_path!s}",
            "output.experiment_name=pytest_smoke",
            "output.plot_grid_size=12",
        ],
    )
    output = run_experiment(config)
    assert (output / "resolved_config.yaml").exists()
    assert (output / "calibration.json").exists()
    for method in ("B1", "A4"):
        run_dir = output / method
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "samples.pt").exists()
        metrics = pd.read_csv(run_dir / "metrics.csv")
        assert "axis/ceq_per_chain" in metrics
        assert "distribution/fid_raw" in metrics
        assert "accept/overall_cumulative" in metrics
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert "final_cost_per_chain" in summary
        samples = torch.load(run_dir / "samples.pt", weights_only=False)
        assert samples["final_samples"].shape == (4, 2)
