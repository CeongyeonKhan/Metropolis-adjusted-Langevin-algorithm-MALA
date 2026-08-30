from __future__ import annotations

import numpy as np
import torch

from ege_ah_mala.diagnostics import (
    bulk_tail_ess,
    kaplan_meier,
    rank_normalized_split_rhat,
    raw_frechet_distance,
)


def test_constant_but_different_chains_have_infinite_rhat() -> None:
    values = np.zeros((4, 20, 1), dtype=float)
    values[2:, :, 0] = 1.0
    rhat = rank_normalized_split_rhat(values)
    assert np.isinf(rhat[0])


def test_identical_constant_chains_have_unit_rhat() -> None:
    values = np.ones((4, 20, 1), dtype=float)
    rhat = rank_normalized_split_rhat(values)
    assert rhat[0] == 1.0


def test_independent_draws_have_finite_ess() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(4, 100, 2))
    bulk, tail = bulk_tail_ess(values)
    assert np.isfinite(bulk).all()
    assert np.isfinite(tail).all()
    assert (bulk > 1.0).all()


def test_raw_frechet_zero_for_matching_empirical_moments() -> None:
    samples = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    true_mean = torch.tensor([0.0], dtype=torch.float64)
    true_covariance = torch.tensor([[2.0]], dtype=torch.float64)
    value = raw_frechet_distance(samples, true_mean, true_covariance)
    torch.testing.assert_close(
        value, torch.tensor(0.0, dtype=torch.float64), atol=1.0e-12, rtol=0.0
    )


def test_kaplan_meier_handles_right_censoring() -> None:
    result = kaplan_meier(
        np.array([1.0, 2.0, 4.0, 4.0]),
        np.array([True, True, False, False]),
        cutoff=4.0,
    )
    assert result["survival"][-1] == 0.5
    assert 0.0 < result["rmst"] <= 4.0
