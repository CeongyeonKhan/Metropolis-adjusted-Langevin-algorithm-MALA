from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .calibration import BasinNormalizer, ErrorCalibrator
from .config import AdaptationConfig
from .error_fields import GradientEnsemble
from .methods import MethodSpec

CONSTRAINT_NAMES = (
    "fixed_base",
    "energy",
    "curvature",
    "gradient_error",
    "drift",
    "hmax",
)


@dataclass
class ProposalState:
    x: torch.Tensor
    energy: torch.Tensor
    gradient: torch.Tensor
    disagreement: torch.Tensor
    true_error_squared: torch.Tensor
    proxy_error_squared: torch.Tensor
    active_error_squared: torch.Tensor
    z_energy: torch.Tensor
    z_gradient: torch.Tensor
    tau: torch.Tensor
    drift_multiplier: torch.Tensor
    h: torch.Tensor
    effective_step: torch.Tensor
    drift_scale: torch.Tensor
    h_fixed: torch.Tensor
    h_energy: torch.Tensor
    h_curvature: torch.Tensor
    h_error: torch.Tensor
    h_drift: torch.Tensor
    active_constraint: torch.Tensor
    numeric_floor: torch.Tensor
    below_operational: torch.Tensor
    curvature: torch.Tensor
    finite: torch.Tensor


class StepController:
    def __init__(
        self,
        ensemble: GradientEnsemble,
        normalizer: BasinNormalizer,
        calibrator: ErrorCalibrator,
        config: AdaptationConfig,
    ) -> None:
        self.ensemble = ensemble
        self.normalizer = normalizer
        self.calibrator = calibrator
        self.config = config

    def evaluate(
        self,
        x: torch.Tensor,
        energy: torch.Tensor,
        method: MethodSpec,
    ) -> ProposalState:
        members = self.ensemble.members(x)
        gradient = members.mean(dim=1)
        disagreement = self.ensemble.disagreement_variance(x, members)
        true_error_squared = self.ensemble.true_error_squared(x, gradient)
        proxy_error_squared = self.calibrator.predict_squared(disagreement)
        z_energy = self.normalizer.z_energy(x, energy)
        z_gradient = self.normalizer.z_gradient(x, gradient)

        if method.adaptive_noise:
            trapped = torch.sigmoid(-self.config.beta_energy * z_energy) * torch.sigmoid(
                -self.config.beta_gradient * z_gradient
            )
            tau = 1.0 + (self.config.tau_max - 1.0) * trapped
        else:
            tau = torch.ones_like(energy)
        drift_multiplier = tau.pow(self.config.gamma)

        if method.energy_adaptation:
            h_fixed = torch.full_like(energy, torch.inf)
            h_energy = self.config.h0 * torch.exp(self.config.alpha_energy * z_energy)
        else:
            h_fixed = torch.full_like(energy, self.config.h0)
            h_energy = torch.full_like(energy, torch.inf)

        if method.curvature_constraint:
            if self.config.curvature_mode == "power" and self.ensemble.config.kind != "orthogonal":
                curvature = self._power_curvature(x)
            else:
                jacobian = self.ensemble.jacobian_mean(x)
                curvature = torch.linalg.svdvals(jacobian)[..., 0]
            curvature = curvature * self.config.curvature_safety
            h_curvature = self.config.curvature_c / (
                drift_multiplier * (curvature + self.config.epsilon)
            )
        else:
            curvature = torch.zeros_like(energy)
            h_curvature = torch.full_like(energy, torch.inf)

        if method.error_constraint == "oracle":
            active_error_squared = true_error_squared
        elif method.error_constraint == "proxy":
            active_error_squared = proxy_error_squared
        else:
            active_error_squared = torch.zeros_like(energy)
        if method.error_constraint == "none":
            h_error = torch.full_like(energy, torch.inf)
        else:
            h_error = (
                4.0
                * self.config.epsilon_proposal
                * tau
                / (drift_multiplier.square() * (active_error_squared + self.config.epsilon))
            )

        if method.drift_constraint:
            radius = self.config.drift_radius_factor * math.sqrt(self.ensemble.target.dim)
            h_drift = radius / (
                drift_multiplier
                * (torch.linalg.vector_norm(gradient, dim=-1) + self.config.epsilon)
            )
        else:
            h_drift = torch.full_like(energy, torch.inf)

        h_max = torch.full_like(energy, self.config.h_max)
        candidates = torch.stack([h_fixed, h_energy, h_curvature, h_error, h_drift, h_max], dim=-1)
        unconstrained_h, active_constraint = torch.min(candidates, dim=-1)
        numeric_floor = unconstrained_h < self.config.h_num
        h = unconstrained_h.clamp_min(self.config.h_num)
        below_operational = unconstrained_h < self.config.h_oper
        effective_step = h * tau
        drift_scale = h * drift_multiplier
        finite = (
            torch.isfinite(energy)
            & torch.isfinite(gradient).all(dim=-1)
            & torch.isfinite(h)
            & torch.isfinite(tau)
            & (h > 0.0)
            & (tau > 0.0)
        )
        return ProposalState(
            x=x,
            energy=energy,
            gradient=gradient,
            disagreement=disagreement,
            true_error_squared=true_error_squared,
            proxy_error_squared=proxy_error_squared,
            active_error_squared=active_error_squared,
            z_energy=z_energy,
            z_gradient=z_gradient,
            tau=tau,
            drift_multiplier=drift_multiplier,
            h=h,
            effective_step=effective_step,
            drift_scale=drift_scale,
            h_fixed=h_fixed,
            h_energy=h_energy,
            h_curvature=h_curvature,
            h_error=h_error,
            h_drift=h_drift,
            active_constraint=active_constraint,
            numeric_floor=numeric_floor,
            below_operational=below_operational,
            curvature=curvature,
            finite=finite,
        )

    def _power_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """以固定探针和固定迭代次数估计保守梯度场的谱范数。"""
        dimension = x.shape[-1]
        coordinate = torch.arange(dimension, dtype=x.dtype, device=x.device) + 0.5
        estimates = []
        for probe in range(self.config.curvature_probes):
            vector = torch.cos(math.pi * (probe + 1.0) * coordinate / max(1, dimension))
            vector = vector / torch.linalg.vector_norm(vector).clamp_min(self.config.epsilon)
            vector = vector[None, :].expand(x.shape[0], -1)
            for _ in range(self.config.curvature_power_iterations):
                product = self.ensemble.jacobian_vector_product_mean(x, vector)
                norm = torch.linalg.vector_norm(product, dim=-1).clamp_min(self.config.epsilon)
                vector = product / norm[:, None]
            product = self.ensemble.jacobian_vector_product_mean(x, vector)
            estimates.append(torch.linalg.vector_norm(product, dim=-1))
        return torch.stack(estimates, dim=-1).amax(dim=-1)

    def curvature_evaluations_per_point(self, method: MethodSpec) -> int:
        """返回按集成成员折算的海森矩阵—向量积次数。"""
        if not method.curvature_constraint:
            return 0
        if self.config.curvature_mode == "power" and self.ensemble.config.kind != "orthogonal":
            return (
                self.config.curvature_probes
                * (self.config.curvature_power_iterations + 1)
                * self.ensemble.ensemble_size
            )
        return self.ensemble.target.dim * self.ensemble.ensemble_size


def isotropic_log_proposal(destination: torch.Tensor, state: ProposalState) -> torch.Tensor:
    mean = state.x - state.h[:, None] * state.drift_multiplier[:, None] * state.gradient
    variance = 2.0 * state.h * state.tau
    squared_distance = (destination - mean).square().sum(dim=-1)
    return -0.5 * state.x.shape[-1] * torch.log(2.0 * math.pi * variance) - 0.5 * (
        squared_distance / variance
    )
