from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from scipy.stats import spearmanr

from .config import CalibrationConfig
from .distributions import GaussianMixtureTarget
from .error_fields import GradientEnsemble


def generate_calibration_points(
    target: GaussianMixtureTarget,
    config: CalibrationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    n = config.num_points
    n_target = round(0.4 * n)
    n_bridge = round(0.4 * n)
    n_wide = n - n_target - n_bridge
    target_points = target.sample(n_target, generator)

    first = torch.randint(
        target.num_components,
        (n_bridge,),
        device=target.device,
        generator=generator,
    )
    offset = torch.randint(
        1,
        max(2, target.num_components),
        (n_bridge,),
        device=target.device,
        generator=generator,
    )
    second = (first + offset) % target.num_components
    interpolation = torch.rand(
        n_bridge, 1, dtype=target.dtype, device=target.device, generator=generator
    )
    bridge = (1.0 - interpolation) * target.means[first] + interpolation * target.means[second]
    bridge = bridge + 0.25 * torch.randn(
        n_bridge,
        target.dim,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )

    true_mean, true_covariance = target.true_moments()
    wide_cholesky = torch.linalg.cholesky(
        config.wide_scale**2 * true_covariance
        + 1.0e-8 * torch.eye(target.dim, dtype=target.dtype, device=target.device)
    )
    wide_noise = torch.randn(
        n_wide,
        target.dim,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    wide = true_mean + torch.einsum("ij,nj->ni", wide_cholesky, wide_noise)

    points = torch.cat([target_points, bridge, wide], dim=0)
    permutation = torch.randperm(n, device=target.device, generator=generator)
    return points[permutation]


def _weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, quantile: float
) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order].clamp_min(0.0)
    cumulative = torch.cumsum(sorted_weights, dim=0)
    total = cumulative[-1]
    if float(total) <= 0.0:
        return torch.quantile(values, quantile)
    threshold = quantile * total
    index = torch.searchsorted(cumulative, threshold).clamp_max(values.numel() - 1)
    return sorted_values[index]


@dataclass
class BasinNormalizer:
    reference: GaussianMixtureTarget
    energy_median: torch.Tensor
    energy_mad: torch.Tensor
    gradient_median: torch.Tensor
    gradient_mad: torch.Tensor
    z_max: float
    epsilon: float = 1.0e-12
    gradient_offset: float = 1.0e-12

    @classmethod
    def fit(
        cls,
        reference: GaussianMixtureTarget,
        ensemble: GradientEnsemble,
        points: torch.Tensor,
        z_max: float,
        epsilon: float = 1.0e-12,
    ) -> BasinNormalizer:
        with torch.no_grad():
            responsibilities = reference.responsibilities(points)
            energy = ensemble.target.energy(points)
            gradient = ensemble.mean(points)
            grad_rms = torch.linalg.vector_norm(gradient, dim=-1) / math.sqrt(reference.dim)
            log_grad = torch.log(grad_rms + epsilon)
            e_med, e_mad, g_med, g_mad = [], [], [], []
            for component in range(reference.num_components):
                weights = responsibilities[:, component]
                median_e = _weighted_quantile(energy, weights, 0.5)
                median_g = _weighted_quantile(log_grad, weights, 0.5)
                mad_e = _weighted_quantile(torch.abs(energy - median_e), weights, 0.5)
                mad_g = _weighted_quantile(torch.abs(log_grad - median_g), weights, 0.5)
                e_med.append(median_e)
                e_mad.append(mad_e)
                g_med.append(median_g)
                g_mad.append(mad_g)
        return cls(
            reference=reference,
            energy_median=torch.stack(e_med),
            energy_mad=torch.stack(e_mad),
            gradient_median=torch.stack(g_med),
            gradient_mad=torch.stack(g_mad),
            z_max=z_max,
            epsilon=epsilon,
            gradient_offset=epsilon,
        )

    def z_energy(self, x: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        responsibilities = self.reference.responsibilities(x)
        scale = 1.4826 * self.energy_mad + self.epsilon
        per_basin = ((energy[:, None] - self.energy_median[None, :]) / scale[None, :]).clamp(
            -self.z_max, self.z_max
        )
        return torch.sum(responsibilities * per_basin, dim=-1)

    def z_gradient(self, x: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        responsibilities = self.reference.responsibilities(x)
        grad_rms = torch.linalg.vector_norm(gradient, dim=-1) / math.sqrt(self.reference.dim)
        log_grad = torch.log(grad_rms + self.gradient_offset)
        scale = 1.4826 * self.gradient_mad + self.epsilon
        per_basin = ((log_grad[:, None] - self.gradient_median[None, :]) / scale[None, :]).clamp(
            -self.z_max, self.z_max
        )
        return torch.sum(responsibilities * per_basin, dim=-1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy_median": self.energy_median.detach().cpu().tolist(),
            "energy_mad": self.energy_mad.detach().cpu().tolist(),
            "gradient_log_median": self.gradient_median.detach().cpu().tolist(),
            "gradient_log_mad": self.gradient_mad.detach().cpu().tolist(),
            "z_max": self.z_max,
        }


def _nnls_two_columns(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    unconstrained = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    candidates = [unconstrained.clamp_min(0.0)]
    candidates.append(torch.stack([torch.zeros_like(x[0]), y.mean().clamp_min(0.0)]))
    slope = torch.sum(x * y) / torch.sum(x.square()).clamp_min(1.0e-20)
    candidates.append(torch.stack([slope.clamp_min(0.0), torch.zeros_like(x[0])]))
    losses = [torch.mean((design @ candidate - y).square()) for candidate in candidates]
    selected = candidates[int(torch.argmin(torch.stack(losses)))]
    return selected[0], selected[1]


@dataclass
class ErrorCalibrator:
    slope: torch.Tensor
    intercept: torch.Tensor
    inflation: torch.Tensor
    epsilon: float
    nominal_coverage: float

    @classmethod
    def fit(
        cls,
        disagreement: torch.Tensor,
        true_error_squared: torch.Tensor,
        config: CalibrationConfig,
        epsilon: float = 1.0e-12,
    ) -> tuple[ErrorCalibrator, dict[str, Any]]:
        n = disagreement.numel()
        n_fit = max(2, int(config.fit_fraction * n))
        n_cal = max(2, int(config.calibration_fraction * n))
        if n_fit + n_cal >= n:
            n_cal = max(1, n - n_fit - 1)
        fit_u = disagreement[:n_fit]
        fit_y = true_error_squared[:n_fit]
        slope, intercept = _nnls_two_columns(fit_u, fit_y)
        cal_u = disagreement[n_fit : n_fit + n_cal]
        cal_y = true_error_squared[n_fit : n_fit + n_cal]
        base = (slope * cal_u + intercept).clamp_min(epsilon)
        scores = cal_y / base
        rank = min(scores.numel(), math.ceil((scores.numel() + 1) * config.coverage))
        inflation = torch.sort(scores).values[rank - 1].clamp_min(1.0)
        calibrator = cls(slope, intercept, inflation, epsilon, config.coverage)
        metadata = {
            "fit_size": n_fit,
            "calibration_size": n_cal,
            "test_size": n - n_fit - n_cal,
            "slope": float(slope.detach().cpu()),
            "intercept": float(intercept.detach().cpu()),
            "inflation": float(inflation.detach().cpu()),
            "nominal_coverage": config.coverage,
            "calibration_definition": "finite_sample_ratio_quantile",
        }
        return calibrator, metadata

    def predict_squared(self, disagreement: torch.Tensor) -> torch.Tensor:
        base = (self.slope * disagreement + self.intercept).clamp_min(self.epsilon)
        return self.inflation * base

    def evaluate(
        self,
        disagreement: torch.Tensor,
        true_error_squared: torch.Tensor,
        responsibilities: torch.Tensor,
        z_energy: torch.Tensor,
    ) -> dict[str, Any]:
        predicted = self.predict_squared(disagreement)
        covered = true_error_squared <= predicted
        max_resp = responsibilities.max(dim=-1).values
        regions = {
            "overall": torch.ones_like(covered, dtype=torch.bool),
            "mode_interior": max_resp >= 0.9,
            "mode_boundary": max_resp < 0.7,
            "high_energy": z_energy > 1.0,
        }
        predicted_numpy = predicted.detach().cpu().numpy()
        true_numpy = true_error_squared.detach().cpu().numpy()
        if predicted_numpy.std() <= self.epsilon or true_numpy.std() <= self.epsilon:
            correlation = float("nan")
        else:
            correlation = float(spearmanr(predicted_numpy, true_numpy).statistic)
        output: dict[str, Any] = {
            "coverage": float(covered.double().mean().cpu()),
            "violation_rate": float((~covered).double().mean().cpu()),
            "spearman": correlation,
        }
        for name, mask in regions.items():
            output[f"coverage_{name}"] = (
                float(covered[mask].double().mean().cpu()) if bool(mask.any()) else float("nan")
            )
        quantile_index = torch.argsort(predicted)
        bins: list[dict[str, float | int]] = []
        for bin_id, indices in enumerate(torch.tensor_split(quantile_index, 5)):
            if indices.numel() == 0:
                continue
            bins.append(
                {
                    "quintile": bin_id + 1,
                    "predicted_delta2": float(predicted[indices].mean().cpu()),
                    "empirical_delta2": float(true_error_squared[indices].mean().cpu()),
                    "coverage": float(covered[indices].double().mean().cpu()),
                    "n": int(indices.numel()),
                }
            )
        output["quintile_bins"] = bins
        return output
