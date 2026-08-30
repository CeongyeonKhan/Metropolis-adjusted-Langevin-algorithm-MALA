from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import norm, rankdata

from .config import MetricsConfig
from .distributions import GaussianMixtureTarget


def _split_chains(values: np.ndarray) -> np.ndarray:
    _, draws, variables = values.shape
    half = draws // 2
    if half < 2:
        return np.empty((0, 0, variables), dtype=float)
    first = values[:, :half, :]
    second = values[:, draws - half :, :]
    return np.concatenate([first, second], axis=0)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    ranks = rankdata(flat, method="average")
    transformed = norm.ppf((ranks - 0.375) / (flat.size + 0.25))
    return transformed.reshape(values.shape)


def _basic_rhat(values: np.ndarray) -> float:
    chains, draws = values.shape
    if chains < 2 or draws < 2:
        return float("nan")
    chain_variances = np.var(values, axis=1, ddof=1)
    within = float(np.mean(chain_variances))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1))
    if within <= 1.0e-30:
        return float("inf") if between > 1.0e-30 else 1.0
    variance = ((draws - 1.0) / draws) * within + between / draws
    return float(math.sqrt(max(variance / within, 0.0)))


def rank_normalized_split_rhat(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[:, :, None]
    split = _split_chains(array)
    if split.shape[1] < 2:
        return np.full(array.shape[-1], np.nan)
    output = []
    for variable in range(split.shape[-1]):
        raw = split[:, :, variable]
        within_constant = np.all(np.ptp(raw, axis=1) <= 1.0e-15)
        if within_constant:
            output.append(float("inf") if np.ptp(np.mean(raw, axis=1)) > 1.0e-15 else 1.0)
            continue
        rank_values = _rank_normalize(raw)
        folded = np.abs(raw - np.median(raw))
        folded_rank = _rank_normalize(folded)
        output.append(max(_basic_rhat(rank_values), _basic_rhat(folded_rank)))
    return np.asarray(output)


def _autocovariance(values: np.ndarray) -> np.ndarray:
    n = values.size
    centered = values - np.mean(values)
    size = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(centered, n=size)
    autocov = np.fft.irfft(fft * np.conjugate(fft), n=size)[:n]
    return autocov / np.arange(n, 0, -1)


def _ess_one(values: np.ndarray) -> float:
    chains, draws = values.shape
    if chains < 2 or draws < 3:
        return float("nan")
    chain_var = np.var(values, axis=1, ddof=1)
    within = float(np.mean(chain_var))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1))
    variance_plus = ((draws - 1.0) / draws) * within + between / draws
    if variance_plus <= 1.0e-30:
        return float(chains * draws)
    mean_autocov = np.mean(np.stack([_autocovariance(chain) for chain in values]), axis=0)
    rho = np.ones(draws)
    rho[1:] = 1.0 - (within - mean_autocov[1:]) / variance_plus
    paired = []
    lag = 0
    while 2 * lag + 1 < draws:
        pair = rho[2 * lag] + rho[2 * lag + 1]
        if pair < 0:
            break
        paired.append(pair)
        lag += 1
    if not paired:
        return float(chains * draws)
    for index in range(1, len(paired)):
        paired[index] = min(paired[index], paired[index - 1])
    tau = -1.0 + 2.0 * float(np.sum(paired))
    ess = chains * draws / max(tau, 1.0)
    return float(min(chains * draws, max(1.0, ess)))


def bulk_tail_ess(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[:, :, None]
    split = _split_chains(array)
    if split.shape[1] < 3:
        shape = (array.shape[-1],)
        return np.full(shape, np.nan), np.full(shape, np.nan)
    bulk, tail = [], []
    for variable in range(split.shape[-1]):
        raw = split[:, :, variable]
        ranked = _rank_normalize(raw)
        bulk.append(_ess_one(ranked))
        lower, upper = np.quantile(raw, [0.05, 0.95])
        lower_indicator = (raw <= lower).astype(float)
        upper_indicator = (raw >= upper).astype(float)
        tail.append(min(_ess_one(lower_indicator), _ess_one(upper_indicator)))
    return np.asarray(bulk), np.asarray(tail)


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp_min(0.0)
    return (eigenvectors * torch.sqrt(eigenvalues)[None, :]) @ eigenvectors.transpose(-1, -2)


def raw_frechet_distance(
    samples: torch.Tensor, true_mean: torch.Tensor, true_covariance: torch.Tensor
) -> torch.Tensor:
    sample_mean = samples.mean(dim=0)
    centered = samples - sample_mean
    denominator = max(1, samples.shape[0] - 1)
    sample_covariance = centered.transpose(0, 1) @ centered / denominator
    true_sqrt = _matrix_sqrt_psd(true_covariance)
    middle = true_sqrt @ sample_covariance @ true_sqrt
    middle_sqrt = _matrix_sqrt_psd(middle)
    distance = (sample_mean - true_mean).square().sum()
    distance = distance + torch.trace(sample_covariance + true_covariance - 2.0 * middle_sqrt)
    return distance.clamp_min(0.0)


def mode_occupancy_metrics(modes: torch.Tensor, weights: torch.Tensor) -> dict[str, float]:
    counts = torch.bincount(modes, minlength=weights.numel()).to(weights.dtype)
    probability = counts / counts.sum().clamp_min(1.0)
    midpoint = 0.5 * (probability + weights)
    kl_p = torch.sum(
        torch.where(probability > 0, probability * torch.log(probability / midpoint), 0.0)
    )
    kl_w = torch.sum(weights * torch.log(weights / midpoint.clamp_min(1.0e-30)))
    js = 0.5 * (kl_p + kl_w)
    ratio = probability / weights
    return {
        "distribution/js": float(js.detach().cpu()),
        "distribution/min_mode_ratio": float(ratio.min().detach().cpu()),
        "distribution/mode_occupancy": probability.detach().cpu().tolist(),
    }


def fid_reference_band(
    target: GaussianMixtureTarget,
    sample_size: int,
    repeats: int,
    generator: torch.Generator,
) -> dict[str, float]:
    true_mean, true_covariance = target.true_moments()
    values = []
    for _ in range(repeats):
        sample = target.sample(sample_size, generator)
        values.append(raw_frechet_distance(sample, true_mean, true_covariance))
    tensor = torch.stack(values)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.05, 0.5, 0.95], dtype=tensor.dtype, device=tensor.device)
    )
    return {
        "distribution/fid_reference_p05": float(quantiles[0].cpu()),
        "distribution/fid_reference_p50": float(quantiles[1].cpu()),
        "distribution/fid_reference_p95": float(quantiles[2].cpu()),
    }


def checkpoint_diagnostics(
    history: torch.Tensor,
    current: torch.Tensor,
    target: GaussianMixtureTarget,
    metrics_config: MetricsConfig,
    reference_band: dict[str, float],
    fid_current: torch.Tensor | None = None,
    fid_target: GaussianMixtureTarget | None = None,
) -> dict[str, Any]:
    active_fid_target = target if fid_target is None else fid_target
    active_fid_current = current if fid_current is None else fid_current
    true_mean, true_covariance = active_fid_target.true_moments()
    output: dict[str, Any] = {
        "distribution/fid_raw": float(
            raw_frechet_distance(active_fid_current, true_mean, true_covariance).detach().cpu()
        )
    }
    output.update(reference_band)
    current_modes = target.mode(current)
    output.update(mode_occupancy_metrics(current_modes, target.weights))

    draws = history.shape[0]
    if draws < metrics_config.min_rhat_draws:
        output.update(
            {
                "diagnostics/rhat_coord_max": float("nan"),
                "diagnostics/rhat_energy": float("nan"),
                "diagnostics/rhat_mode_max": float("nan"),
                "diagnostics/rhat_max": float("nan"),
                "diagnostics/rhat_has_infinite": 0,
                "diagnostics/ess_bulk_min": float("nan"),
                "diagnostics/ess_tail_min": float("nan"),
                "diagnostics/ess_coord_bulk_min": float("nan"),
                "diagnostics/ess_coord_tail_min": float("nan"),
                "diagnostics/ess_energy_bulk": float("nan"),
                "diagnostics/ess_energy_tail": float("nan"),
                "diagnostics/ess_mode_bulk_min": float("nan"),
                "diagnostics/ess_mode_tail_min": float("nan"),
            }
        )
        return output

    chains_first = history.permute(1, 0, 2).detach()
    flat = history.reshape(-1, target.dim)
    energy = target.energy(flat).reshape(draws, history.shape[1]).transpose(0, 1)
    responsibilities = (
        target.responsibilities(flat)
        .reshape(draws, history.shape[1], target.num_components)
        .permute(1, 0, 2)
    )
    hard_modes = responsibilities.argmax(dim=-1)
    indicators = torch.nn.functional.one_hot(hard_modes, num_classes=target.num_components).to(
        history.dtype
    )

    coordinates_np = chains_first.cpu().numpy()
    energy_np = energy[:, :, None].cpu().numpy()
    mode_features_np = torch.cat([responsibilities, indicators], dim=-1).cpu().numpy()
    rhat_coord = rank_normalized_split_rhat(coordinates_np)
    rhat_energy = rank_normalized_split_rhat(energy_np)
    rhat_mode = rank_normalized_split_rhat(mode_features_np)
    ess_coord_bulk, ess_coord_tail = bulk_tail_ess(coordinates_np)
    ess_energy_bulk, ess_energy_tail = bulk_tail_ess(energy_np)
    ess_mode_bulk, ess_mode_tail = bulk_tail_ess(mode_features_np)
    ess_bulk = np.concatenate([ess_coord_bulk, ess_energy_bulk, ess_mode_bulk])
    ess_tail = np.concatenate([ess_coord_tail, ess_energy_tail, ess_mode_tail])
    rhat_all = np.concatenate([rhat_coord, rhat_energy, rhat_mode])
    output.update(
        {
            "diagnostics/rhat_coord_max": float(np.nanmax(rhat_coord)),
            "diagnostics/rhat_energy": float(rhat_energy[0]),
            "diagnostics/rhat_mode_max": float(np.nanmax(rhat_mode)),
            "diagnostics/rhat_max": float(np.nanmax(rhat_all)),
            "diagnostics/rhat_has_infinite": int(np.isinf(rhat_all).any()),
            "diagnostics/ess_bulk_min": float(np.nanmin(ess_bulk)),
            "diagnostics/ess_tail_min": float(np.nanmin(ess_tail)),
            "diagnostics/ess_coord_bulk_min": float(np.nanmin(ess_coord_bulk)),
            "diagnostics/ess_coord_tail_min": float(np.nanmin(ess_coord_tail)),
            "diagnostics/ess_energy_bulk": float(ess_energy_bulk[0]),
            "diagnostics/ess_energy_tail": float(ess_energy_tail[0]),
            "diagnostics/ess_mode_bulk_min": float(np.nanmin(ess_mode_bulk)),
            "diagnostics/ess_mode_tail_min": float(np.nanmin(ess_mode_tail)),
        }
    )
    return output


@dataclass
class ModeCrossingTracker:
    initial_mode: torch.Tensor
    current_mode: torch.Tensor
    visited: torch.Tensor
    confidence: float
    confirmation: int
    pending_mode: torch.Tensor
    pending_count: torch.Tensor
    pending_start_cost: torch.Tensor
    first_cross_cost: torch.Tensor
    first_cross_confirmation_cost: torch.Tensor
    first_half_cost: torch.Tensor
    first_all_cost: torch.Tensor
    transition_matrix: torch.Tensor
    switches: int = 0

    @classmethod
    def create(
        cls,
        responsibilities: torch.Tensor,
        confidence: float,
        confirmation: int,
    ) -> ModeCrossingTracker:
        modes = responsibilities.argmax(dim=-1)
        chains, components = responsibilities.shape
        visited = torch.zeros(chains, components, dtype=torch.bool, device=responsibilities.device)
        visited[torch.arange(chains, device=responsibilities.device), modes] = True
        nan_cost = torch.full(
            (chains,), float("nan"), dtype=responsibilities.dtype, device=responsibilities.device
        )
        half = math.ceil(components / 2)
        first_half = torch.where(
            visited.sum(dim=1) >= half,
            torch.zeros_like(nan_cost),
            nan_cost,
        )
        first_all = torch.where(
            visited.sum(dim=1) == components,
            torch.zeros_like(nan_cost),
            nan_cost,
        )
        return cls(
            initial_mode=modes.clone(),
            current_mode=modes.clone(),
            visited=visited,
            confidence=confidence,
            confirmation=confirmation,
            pending_mode=torch.full_like(modes, -1),
            pending_count=torch.zeros_like(modes),
            pending_start_cost=nan_cost.clone(),
            first_cross_cost=nan_cost.clone(),
            first_cross_confirmation_cost=nan_cost.clone(),
            first_half_cost=first_half,
            first_all_cost=first_all,
            transition_matrix=torch.zeros(
                components, components, dtype=torch.long, device=responsibilities.device
            ),
        )

    def update(
        self,
        responsibilities: torch.Tensor,
        cost_per_chain: float | torch.Tensor,
    ) -> None:
        if isinstance(cost_per_chain, torch.Tensor):
            costs = cost_per_chain.to(dtype=responsibilities.dtype, device=responsibilities.device)
            if costs.ndim == 0:
                costs = costs.expand(responsibilities.shape[0])
            if costs.shape != (responsibilities.shape[0],):
                raise ValueError("逐链成本张量形状必须为 [num_chains]")
        else:
            costs = torch.full(
                (responsibilities.shape[0],),
                float(cost_per_chain),
                dtype=responsibilities.dtype,
                device=responsibilities.device,
            )
        confidence, proposed = responsibilities.max(dim=-1)
        for chain in range(responsibilities.shape[0]):
            if float(confidence[chain]) < self.confidence:
                self.pending_mode[chain] = -1
                self.pending_count[chain] = 0
                self.pending_start_cost[chain] = float("nan")
                continue
            new_mode = proposed[chain]
            if int(new_mode) == int(self.current_mode[chain]):
                self.pending_mode[chain] = -1
                self.pending_count[chain] = 0
                self.pending_start_cost[chain] = float("nan")
                continue
            if int(self.pending_mode[chain]) == int(new_mode):
                self.pending_count[chain] += 1
            else:
                self.pending_mode[chain] = new_mode
                self.pending_count[chain] = 1
                self.pending_start_cost[chain] = costs[chain]
            if int(self.pending_count[chain]) < self.confirmation:
                continue
            old_mode = int(self.current_mode[chain])
            confirmed_mode = int(new_mode)
            self.transition_matrix[old_mode, confirmed_mode] += 1
            self.current_mode[chain] = confirmed_mode
            self.visited[chain, confirmed_mode] = True
            self.switches += 1
            if not bool(torch.isfinite(self.first_cross_cost[chain])) and confirmed_mode != int(
                self.initial_mode[chain]
            ):
                self.first_cross_cost[chain] = self.pending_start_cost[chain]
                self.first_cross_confirmation_cost[chain] = costs[chain]
            visited_count = int(self.visited[chain].sum())
            half = math.ceil(self.visited.shape[1] / 2)
            if visited_count >= half and not bool(torch.isfinite(self.first_half_cost[chain])):
                self.first_half_cost[chain] = self.pending_start_cost[chain]
            if visited_count == self.visited.shape[1] and not bool(
                torch.isfinite(self.first_all_cost[chain])
            ):
                self.first_all_cost[chain] = self.pending_start_cost[chain]
            self.pending_mode[chain] = -1
            self.pending_count[chain] = 0
            self.pending_start_cost[chain] = float("nan")

    def _cutoffs(self, cutoff: float | torch.Tensor) -> torch.Tensor:
        if isinstance(cutoff, torch.Tensor):
            values = cutoff.to(
                dtype=self.first_cross_cost.dtype, device=self.first_cross_cost.device
            )
            if values.ndim == 0:
                values = values.expand_as(self.first_cross_cost)
            if values.shape != self.first_cross_cost.shape:
                raise ValueError("删失成本张量形状必须为 [num_chains]")
            return values
        return torch.full_like(self.first_cross_cost, float(cutoff))

    @staticmethod
    def _finite_median(values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        return float(torch.median(finite).cpu()) if finite.numel() else float("nan")

    def summary(self, cutoff: float | torch.Tensor) -> dict[str, float | int]:
        cutoffs = self._cutoffs(cutoff)
        common_cutoff = float(torch.min(cutoffs).cpu())
        events = torch.isfinite(self.first_cross_cost)
        events = events & (self.first_cross_cost <= common_cutoff)
        observed = torch.where(
            events,
            self.first_cross_cost,
            torch.full_like(self.first_cross_cost, common_cutoff),
        )
        curve = kaplan_meier(
            observed.detach().cpu().numpy(), events.detach().cpu().numpy(), common_cutoff
        )
        return {
            "crossing/censored_fraction": float((~events).double().mean().cpu()),
            "crossing/rmst_cost": float(curve["rmst"]),
            "crossing/switches_per_1000_cost": float(
                1000.0 * self.switches / max(1.0, float(cutoffs.sum().cpu()))
            ),
            "crossing/confirmed_switches": self.switches,
            "crossing/first_half_median_cost": self._finite_median(self.first_half_cost),
            "crossing/first_all_median_cost": self._finite_median(self.first_all_cost),
            "crossing/first_half_censored_fraction": float(
                (~torch.isfinite(self.first_half_cost)).double().mean().cpu()
            ),
            "crossing/first_all_censored_fraction": float(
                (~torch.isfinite(self.first_all_cost)).double().mean().cpu()
            ),
        }

    def survival_data(self, cutoff: float | torch.Tensor) -> dict[str, Any]:
        cutoffs = self._cutoffs(cutoff)
        common_cutoff = float(torch.min(cutoffs).cpu())
        events = torch.isfinite(self.first_cross_cost)
        events = events & (self.first_cross_cost <= cutoffs)
        observed = torch.where(events, self.first_cross_cost, cutoffs)
        common_events = events & (self.first_cross_cost <= common_cutoff)
        common_observed = torch.where(
            common_events,
            self.first_cross_cost,
            torch.full_like(self.first_cross_cost, common_cutoff),
        )
        output = kaplan_meier(
            common_observed.detach().cpu().numpy(),
            common_events.detach().cpu().numpy(),
            common_cutoff,
        )
        output.update(
            {
                "observed_time": observed.detach().cpu().tolist(),
                "event": events.detach().cpu().tolist(),
                "censored": (~events).detach().cpu().tolist(),
                "initial_mode": self.initial_mode.detach().cpu().tolist(),
                "confirmation_time": self.first_cross_confirmation_cost.detach().cpu().tolist(),
                "first_half": self.first_half_cost.detach().cpu().tolist(),
                "first_all": self.first_all_cost.detach().cpu().tolist(),
                "chain_cutoff": cutoffs.detach().cpu().tolist(),
                "common_cutoff": common_cutoff,
            }
        )
        return output


def kaplan_meier(times: np.ndarray, events: np.ndarray, cutoff: float) -> dict[str, Any]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    event_times = np.unique(times[events])
    survival = 1.0
    curve_times = [0.0]
    curve_survival = [1.0]
    rmst = 0.0
    previous = 0.0
    for time in event_times:
        if time > cutoff:
            break
        rmst += survival * (time - previous)
        at_risk = int(np.sum(times >= time))
        deaths = int(np.sum((times == time) & events))
        if at_risk > 0:
            survival *= 1.0 - deaths / at_risk
        curve_times.append(float(time))
        curve_survival.append(float(survival))
        previous = float(time)
    rmst += survival * max(0.0, cutoff - previous)
    if curve_times[-1] < cutoff:
        curve_times.append(float(cutoff))
        curve_survival.append(float(survival))
    return {"time": curve_times, "survival": curve_survival, "rmst": float(rmst)}
