from __future__ import annotations

import torch

from ege_ah_mala.calibration import ErrorCalibrator
from ege_ah_mala.config import CalibrationConfig


def test_ratio_calibration_uses_independent_split_and_nonnegative_parameters() -> None:
    disagreement = torch.linspace(0.0, 2.0, 200, dtype=torch.float64)
    true_error = 1.5 * disagreement + 0.2
    true_error[100:150] *= 1.2
    config = CalibrationConfig(
        num_points=200,
        fit_fraction=0.5,
        calibration_fraction=0.25,
        coverage=0.95,
    )
    calibrator, metadata = ErrorCalibrator.fit(disagreement, true_error, config)
    assert float(calibrator.slope) >= 0.0
    assert float(calibrator.intercept) >= 0.0
    assert float(calibrator.inflation) >= 1.0
    assert metadata["fit_size"] == 100
    assert metadata["calibration_size"] == 50
    predicted = calibrator.predict_squared(disagreement[150:])
    assert bool(torch.isfinite(predicted).all())
