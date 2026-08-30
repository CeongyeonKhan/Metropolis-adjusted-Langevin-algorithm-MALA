from __future__ import annotations

import torch

from ege_ah_mala.config import TargetConfig, WhiteningConfig
from ege_ah_mala.distributions import build_target
from ege_ah_mala.whitening import AffineWhitening


def test_whitening_round_trip_and_pooled_covariance() -> None:
    target = build_target(
        TargetConfig(family="bimodal", dim=2, separation=3.0), dtype=torch.float64
    )
    transform = AffineWhitening.fit(
        target,
        WhiteningConfig(enabled=True, source="pooled_component"),
    )
    points = target.sample(16, torch.Generator().manual_seed(9))
    whitened = transform.to_white(points)
    reconstructed = transform.from_white(whitened)
    torch.testing.assert_close(reconstructed, points, rtol=1.0e-10, atol=1.0e-10)
    transformed_target = transform.transform_target(target)
    pooled = torch.einsum("k,kij->ij", transformed_target.weights, transformed_target.covariances)
    torch.testing.assert_close(
        pooled,
        torch.eye(2, dtype=torch.float64),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
