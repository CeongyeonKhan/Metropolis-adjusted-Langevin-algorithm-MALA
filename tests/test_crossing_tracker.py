from __future__ import annotations

import math

import pytest
import torch

from ege_ah_mala.diagnostics import ModeCrossingTracker


def _responsibilities(*rows: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def test_bimodal_initial_state_already_reaches_half_coverage() -> None:
    tracker = ModeCrossingTracker.create(
        _responsibilities((0.95, 0.05), (0.10, 0.90)),
        confidence=0.8,
        confirmation=2,
    )

    torch.testing.assert_close(tracker.first_half_cost, torch.zeros(2, dtype=torch.float64))
    assert bool(torch.isnan(tracker.first_all_cost).all())


def test_confirmation_records_entry_and_confirmation_costs_separately() -> None:
    tracker = ModeCrossingTracker.create(
        _responsibilities((0.95, 0.05)),
        confidence=0.8,
        confirmation=3,
    )
    destination = _responsibilities((0.05, 0.95))

    tracker.update(destination, cost_per_chain=4.0)
    tracker.update(destination, cost_per_chain=7.0)
    assert math.isnan(float(tracker.first_cross_cost[0]))

    tracker.update(destination, cost_per_chain=11.0)

    assert float(tracker.first_cross_cost[0]) == pytest.approx(4.0)
    assert float(tracker.first_cross_confirmation_cost[0]) == pytest.approx(11.0)
    assert tracker.switches == 1
    assert int(tracker.current_mode[0]) == 1


def test_low_confidence_observation_resets_pending_transition() -> None:
    tracker = ModeCrossingTracker.create(
        _responsibilities((0.95, 0.05)),
        confidence=0.8,
        confirmation=2,
    )
    destination = _responsibilities((0.05, 0.95))

    tracker.update(destination, cost_per_chain=5.0)
    assert int(tracker.pending_count[0]) == 1
    assert float(tracker.pending_start_cost[0]) == pytest.approx(5.0)

    tracker.update(_responsibilities((0.50, 0.50)), cost_per_chain=6.0)
    assert int(tracker.pending_mode[0]) == -1
    assert int(tracker.pending_count[0]) == 0
    assert math.isnan(float(tracker.pending_start_cost[0]))

    tracker.update(destination, cost_per_chain=9.0)
    tracker.update(destination, cost_per_chain=10.0)

    assert float(tracker.first_cross_cost[0]) == pytest.approx(9.0)
    assert float(tracker.first_cross_confirmation_cost[0]) == pytest.approx(10.0)


def test_survival_data_exposes_right_censoring_per_chain() -> None:
    tracker = ModeCrossingTracker.create(
        _responsibilities((0.95, 0.05), (0.90, 0.10), (0.05, 0.95)),
        confidence=0.8,
        confirmation=2,
    )

    data = tracker.survival_data(cutoff=42.0)
    summary = tracker.summary(cutoff=42.0)

    assert data["observed_time"] == [42.0, 42.0, 42.0]
    assert data["event"] == [False, False, False]
    assert data["censored"] == [True, True, True]
    assert data["initial_mode"] == [0, 0, 1]
    assert summary["crossing/censored_fraction"] == pytest.approx(1.0)
    assert summary["crossing/rmst_cost"] == pytest.approx(42.0)


def test_update_accepts_chain_specific_cost_tensor() -> None:
    tracker = ModeCrossingTracker.create(
        _responsibilities((0.95, 0.05), (0.90, 0.10)),
        confidence=0.8,
        confirmation=2,
    )
    destination = _responsibilities((0.05, 0.95), (0.10, 0.90))

    tracker.update(destination, cost_per_chain=torch.tensor([2.0, 20.0], dtype=torch.float64))
    tracker.update(destination, cost_per_chain=torch.tensor([3.0, 30.0], dtype=torch.float64))

    torch.testing.assert_close(
        tracker.first_cross_cost,
        torch.tensor([2.0, 20.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tracker.first_cross_confirmation_cost,
        torch.tensor([3.0, 30.0], dtype=torch.float64),
    )
