from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import CostConfig


@dataclass
class CostCounter:
    """同时维护总体成本和逐链实际成本。"""

    num_chains: int | None = None
    device: torch.device | str = "cpu"
    n_energy: int = 0
    n_gradient: int = 0
    n_hessian: int = 0
    n_global_density: int = 0
    n_diffusion: int = 0
    _chain_energy: torch.Tensor | None = field(init=False, default=None, repr=False)
    _chain_gradient: torch.Tensor | None = field(init=False, default=None, repr=False)
    _chain_hessian: torch.Tensor | None = field(init=False, default=None, repr=False)
    _chain_global_density: torch.Tensor | None = field(init=False, default=None, repr=False)
    _chain_diffusion: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.num_chains is None:
            return
        if self.num_chains < 1:
            raise ValueError("num_chains 必须为正整数")
        for name in (
            "_chain_energy",
            "_chain_gradient",
            "_chain_hessian",
            "_chain_global_density",
            "_chain_diffusion",
        ):
            setattr(
                self,
                name,
                torch.zeros(self.num_chains, dtype=torch.float64, device=self.device),
            )

    @staticmethod
    def _increment(
        values: torch.Tensor | None,
        indices: torch.Tensor | None,
        amount: float,
    ) -> None:
        if values is None or indices is None or indices.numel() == 0:
            return
        values.index_add_(
            0,
            indices,
            torch.full(
                (indices.numel(),),
                float(amount),
                dtype=values.dtype,
                device=values.device,
            ),
        )

    def add_energy(self, points: int, indices: torch.Tensor | None = None) -> None:
        self.n_energy += int(points)
        self._increment(self._chain_energy, indices, 1)

    def add_controller(
        self,
        points: int,
        ensemble_size: int,
        curvature_evaluations: int,
        indices: torch.Tensor | None = None,
    ) -> None:
        gradient_count = int(ensemble_size)
        hessian_count = int(curvature_evaluations)
        self.n_gradient += int(points) * gradient_count
        self.n_hessian += int(points) * hessian_count
        self._increment(self._chain_gradient, indices, gradient_count)
        self._increment(self._chain_hessian, indices, hessian_count)

    def add_global_density(
        self,
        evaluations: int,
        indices: torch.Tensor | None = None,
        evaluations_per_chain: int = 1,
    ) -> None:
        self.n_global_density += int(evaluations)
        self._increment(self._chain_global_density, indices, evaluations_per_chain)

    def equivalent(self, weights: CostConfig) -> float:
        return float(
            self.n_gradient
            + weights.hessian_weight * self.n_hessian
            + weights.energy_weight * self.n_energy
            + weights.global_density_weight * self.n_global_density
            + weights.diffusion_weight * self.n_diffusion
        )

    def equivalent_by_chain(self, weights: CostConfig) -> torch.Tensor:
        if self._chain_gradient is None:
            raise RuntimeError("逐链成本仅在初始化 CostCounter(num_chains=...) 后可用")
        assert self._chain_energy is not None
        assert self._chain_hessian is not None
        assert self._chain_global_density is not None
        assert self._chain_diffusion is not None
        return (
            self._chain_gradient
            + weights.hessian_weight * self._chain_hessian
            + weights.energy_weight * self._chain_energy
            + weights.global_density_weight * self._chain_global_density
            + weights.diffusion_weight * self._chain_diffusion
        )

    def as_dict(self, weights: CostConfig, num_chains: int) -> dict[str, float | int]:
        return {
            "n_energy": self.n_energy,
            "n_gradient": self.n_gradient,
            "n_hessian": self.n_hessian,
            "n_global_density": self.n_global_density,
            "n_diffusion": self.n_diffusion,
            "equivalent_total": self.equivalent(weights),
            "equivalent_per_chain": self.equivalent(weights) / num_chains,
        }
