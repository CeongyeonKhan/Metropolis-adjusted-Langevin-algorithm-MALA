from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import ErrorConfig
from .distributions import GaussianMixtureTarget


@dataclass
class GradientEnsemble:
    target: GaussianMixtureTarget
    config: ErrorConfig
    amplitudes: torch.Tensor
    frequencies: torch.Tensor
    phases: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def create(
        cls,
        target: GaussianMixtureTarget,
        config: ErrorConfig,
        scale_points: torch.Tensor,
        generator: torch.Generator,
    ) -> GradientEnsemble:
        j = config.ensemble_size
        m = config.features
        d = target.dim
        amplitudes = torch.randn(
            j, m, dtype=target.dtype, device=target.device, generator=generator
        )
        frequencies = config.frequency_scale * torch.randn(
            j, m, d, dtype=target.dtype, device=target.device, generator=generator
        )
        phases = (
            2.0
            * math.pi
            * torch.rand(j, m, dtype=target.dtype, device=target.device, generator=generator)
        )
        provisional = cls(
            target=target,
            config=config,
            amplitudes=amplitudes,
            frequencies=frequencies,
            phases=phases,
            scale=torch.tensor(1.0, dtype=target.dtype, device=target.device),
        )
        if config.relative_rmse == 0.0 or config.kind.lower() == "exact":
            provisional.scale = torch.tensor(0.0, dtype=target.dtype, device=target.device)
            return provisional
        with torch.no_grad():
            true_gradient = target.grad_energy(scale_points)
            raw_mean = provisional._raw_error(scale_points).mean(dim=1)
            numerator = torch.sqrt(torch.mean(torch.sum(true_gradient.square(), dim=-1)))
            denominator = torch.sqrt(torch.mean(torch.sum(raw_mean.square(), dim=-1))).clamp_min(
                1.0e-12
            )
            provisional.scale = config.relative_rmse * numerator / denominator
        return provisional

    @property
    def ensemble_size(self) -> int:
        return self.config.ensemble_size

    def _fourier_error(self, x: torch.Tensor) -> torch.Tensor:
        phase = torch.einsum("nd,jmd->njm", x, self.frequencies) + self.phases[None, :, :]
        factor = -math.sqrt(2.0 / self.config.features)
        return factor * torch.einsum(
            "njm,jm,jmd->njd", torch.sin(phase), self.amplitudes, self.frequencies
        )

    def _fourier_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        phase = torch.einsum("nd,jmd->njm", x, self.frequencies) + self.phases[None, :, :]
        factor = -math.sqrt(2.0 / self.config.features)
        return factor * torch.einsum(
            "njm,jm,jmi,jml->njil",
            torch.cos(phase),
            self.amplitudes,
            self.frequencies,
            self.frequencies,
        )

    def _fourier_jacobian_vector_product(
        self, x: torch.Tensor, vector: torch.Tensor
    ) -> torch.Tensor:
        phase = torch.einsum("nd,jmd->njm", x, self.frequencies) + self.phases[None, :, :]
        projection = torch.einsum("jmd,nd->njm", self.frequencies, vector)
        factor = -math.sqrt(2.0 / self.config.features)
        return factor * torch.einsum(
            "njm,jm,njm,jmd->njd",
            torch.cos(phase),
            self.amplitudes,
            projection,
            self.frequencies,
        )

    def _rotation(self) -> torch.Tensor:
        rotation = torch.zeros(
            self.target.dim, self.target.dim, dtype=self.target.dtype, device=self.target.device
        )
        if self.target.dim >= 2:
            rotation[0, 1] = -1.0
            rotation[1, 0] = 1.0
        else:
            rotation[0, 0] = -1.0
        return rotation

    def _boundary_multiplier(self, x: torch.Tensor) -> torch.Tensor:
        responsibilities = self.target.responsibilities(x)
        entropy = -torch.sum(
            responsibilities * torch.log(responsibilities.clamp_min(1.0e-12)), dim=-1
        )
        normalizer = math.log(max(2, self.target.num_components))
        return 1.0 + self.config.boundary_gain * entropy / normalizer

    def _raw_error(self, x: torch.Tensor) -> torch.Tensor:
        kind = self.config.kind.lower()
        fourier = self._fourier_error(x)
        if kind in {"fourier", "conservative"}:
            return fourier
        true_gradient = self.target.grad_energy(x)
        if kind == "scale":
            common = self.config.scale_bias * true_gradient[:, None, :]
            return common + 0.05 * fourier
        if kind == "orthogonal":
            rotated = torch.einsum("ij,nj->ni", self._rotation(), true_gradient)
            return rotated[:, None, :] + 0.05 * fourier
        if kind == "boundary":
            return self._boundary_multiplier(x)[:, None, None] * fourier
        if kind == "exact":
            return torch.zeros_like(fourier)
        raise ValueError(f"未知梯度误差类型 kind={self.config.kind}")

    def members(self, x: torch.Tensor) -> torch.Tensor:
        true_gradient = self.target.grad_energy(x)
        return true_gradient[:, None, :] + self.scale * self._raw_error(x)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.members(x).mean(dim=1)

    def disagreement_variance(
        self, x: torch.Tensor, members: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = self.members(x) if members is None else members
        centered = values - values.mean(dim=1, keepdim=True)
        return centered.square().sum(dim=(-1, -2)) / (self.ensemble_size * (self.ensemble_size - 1))

    def true_error_squared(self, x: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
        proposal_gradient = self.mean(x) if mean is None else mean
        return (proposal_gradient - self.target.grad_energy(x)).square().sum(dim=-1)

    def jacobian_mean(self, x: torch.Tensor) -> torch.Tensor:
        target_hessian = self.target.hessian_energy(x)
        fourier_jacobian = self._fourier_jacobian(x)
        kind = self.config.kind.lower()
        if kind in {"fourier", "conservative"}:
            raw = fourier_jacobian
        elif kind == "scale":
            raw = self.config.scale_bias * target_hessian[:, None, :, :] + 0.05 * fourier_jacobian
        elif kind == "orthogonal":
            rotated = torch.einsum("ij,njl->nil", self._rotation(), target_hessian)
            raw = rotated[:, None, :, :] + 0.05 * fourier_jacobian
        elif kind == "boundary":
            # 工程近似：保留冻结的边界倍率，但不把倍率空间导数称为严格上界。
            raw = self._boundary_multiplier(x)[:, None, None, None] * fourier_jacobian
        elif kind == "exact":
            raw = torch.zeros_like(fourier_jacobian)
        else:
            raise ValueError(f"未知梯度误差类型 kind={self.config.kind}")
        return target_hessian + self.scale * raw.mean(dim=1)

    def jacobian_vector_product_mean(self, x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """计算平均梯度场的雅可比—向量积，用于高维固定预算幂迭代。"""
        target_product = self.target.hessian_vector_product(x, vector)
        fourier_product = self._fourier_jacobian_vector_product(x, vector)
        kind = self.config.kind.lower()
        if kind in {"fourier", "conservative"}:
            raw = fourier_product
        elif kind == "scale":
            raw = self.config.scale_bias * target_product[:, None, :] + 0.05 * fourier_product
        elif kind == "orthogonal":
            rotated = torch.einsum("ij,nj->ni", self._rotation(), target_product)
            raw = rotated[:, None, :] + 0.05 * fourier_product
        elif kind == "boundary":
            raw = self._boundary_multiplier(x)[:, None, None] * fourier_product
        elif kind == "exact":
            raw = torch.zeros_like(fourier_product)
        else:
            raise ValueError(f"未知梯度误差类型 kind={self.config.kind}")
        return target_product + self.scale * raw.mean(dim=1)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.config.kind,
            "ensemble_size": self.ensemble_size,
            "features": self.config.features,
            "relative_rmse": self.config.relative_rmse,
            "fitted_scale": float(self.scale.detach().cpu()),
            "frequency_scale": self.config.frequency_scale,
            "boundary_jacobian_is_local_proxy": self.config.kind.lower() == "boundary",
        }
