from __future__ import annotations

import torch

from ege_ah_mala.adaptation import StepController
from ege_ah_mala.calibration import BasinNormalizer, ErrorCalibrator, generate_calibration_points
from ege_ah_mala.config import (
    AdaptationConfig,
    CalibrationConfig,
    ErrorConfig,
    TargetConfig,
)
from ege_ah_mala.distributions import build_target
from ege_ah_mala.error_fields import GradientEnsemble


def make_test_system(gamma: float = 1.0, relative_rmse: float = 0.2):
    dtype = torch.float64
    target = build_target(TargetConfig(family="bimodal", dim=2, separation=2.0), dtype=dtype)
    calibration_config = CalibrationConfig(
        num_points=256,
        fit_fraction=0.5,
        calibration_fraction=0.25,
    )
    point_generator = torch.Generator().manual_seed(11)
    points = generate_calibration_points(target, calibration_config, point_generator)
    ensemble = GradientEnsemble.create(
        target,
        ErrorConfig(
            kind="fourier",
            ensemble_size=5,
            features=12,
            relative_rmse=relative_rmse,
        ),
        points,
        torch.Generator().manual_seed(12),
    )
    normalizer = BasinNormalizer.fit(target, ensemble, points, z_max=3.0)
    members = ensemble.members(points)
    disagreement = ensemble.disagreement_variance(points, members)
    true_error = ensemble.true_error_squared(points, members.mean(dim=1))
    calibrator, _ = ErrorCalibrator.fit(disagreement, true_error, calibration_config)
    adaptation = AdaptationConfig(
        h0=0.04,
        h_max=0.3,
        gamma=gamma,
        tau_max=2.0,
    )
    controller = StepController(ensemble, normalizer, calibrator, adaptation)
    return target, ensemble, controller
