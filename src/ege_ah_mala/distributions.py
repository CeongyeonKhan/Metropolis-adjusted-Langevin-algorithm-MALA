from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .config import TargetConfig


@dataclass
class GaussianMixtureTarget:
    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor

    def __post_init__(self) -> None:
        if self.weights.ndim != 1 or self.means.ndim != 2 or self.covariances.ndim != 3:
            raise ValueError("混合权重、均值和协方差的维数不正确")
        if self.means.shape[0] != self.weights.shape[0]:
            raise ValueError("混合分量数量不一致")
        if self.covariances.shape != (
            self.weights.shape[0],
            self.means.shape[1],
            self.means.shape[1],
        ):
            raise ValueError("协方差形状不正确")
        if torch.any(self.weights <= 0):
            raise ValueError("混合权重必须严格为正")
        self.weights = self.weights / self.weights.sum()
        self.precisions = torch.linalg.inv(self.covariances)
        self.cholesky = torch.linalg.cholesky(self.covariances)
        sign, logdet = torch.linalg.slogdet(self.covariances)
        if torch.any(sign <= 0):
            raise ValueError("协方差必须正定")
        self.logdet = logdet
        self.log_weights = torch.log(self.weights)

    @property
    def num_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    def _flatten(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        if x.shape[-1] != self.dim:
            raise ValueError(f"输入末维应为 {self.dim}，实际为 {x.shape[-1]}")
        leading = x.shape[:-1]
        return x.reshape(-1, self.dim), leading

    def component_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        flat, leading = self._flatten(x)
        diff = flat[:, None, :] - self.means[None, :, :]
        mahal = torch.einsum("nki,kij,nkj->nk", diff, self.precisions, diff)
        normalizer = self.dim * math.log(2.0 * math.pi) + self.logdet
        values = self.log_weights - 0.5 * (normalizer[None, :] + mahal)
        return values.reshape(*leading, self.num_components)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.component_log_prob(x), dim=-1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x)

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.component_log_prob(x), dim=-1)

    def grad_energy(self, x: torch.Tensor) -> torch.Tensor:
        flat, leading = self._flatten(x)
        responsibilities = self.responsibilities(flat)
        diff = flat[:, None, :] - self.means[None, :, :]
        component_grad = torch.einsum("kij,nkj->nki", self.precisions, diff)
        gradient = torch.einsum("nk,nki->ni", responsibilities, component_grad)
        return gradient.reshape(*leading, self.dim)

    def hessian_energy(self, x: torch.Tensor) -> torch.Tensor:
        flat, leading = self._flatten(x)
        responsibilities = self.responsibilities(flat)
        diff = flat[:, None, :] - self.means[None, :, :]
        component_grad = torch.einsum("kij,nkj->nki", self.precisions, diff)
        mean_grad = torch.einsum("nk,nki->ni", responsibilities, component_grad)
        precision_mean = torch.einsum("nk,kij->nij", responsibilities, self.precisions)
        second_moment = torch.einsum(
            "nk,nki,nkj->nij", responsibilities, component_grad, component_grad
        )
        hessian = precision_mean + torch.einsum("ni,nj->nij", mean_grad, mean_grad) - second_moment
        return hessian.reshape(*leading, self.dim, self.dim)

    def hessian_vector_product(self, x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """计算能量海森矩阵—向量积，而不显式构造 ``d × d`` 矩阵。"""
        flat, leading = self._flatten(x)
        flat_vector, vector_leading = self._flatten(vector)
        if vector_leading != leading:
            raise ValueError("x 与 vector 的前导形状必须一致")
        responsibilities = self.responsibilities(flat)
        diff = flat[:, None, :] - self.means[None, :, :]
        component_grad = torch.einsum("kij,nkj->nki", self.precisions, diff)
        mean_grad = torch.einsum("nk,nki->ni", responsibilities, component_grad)
        precision_vector = torch.einsum("kij,nj->nki", self.precisions, flat_vector)
        first = torch.einsum("nk,nki->ni", responsibilities, precision_vector)
        mean_projection = torch.sum(mean_grad * flat_vector, dim=-1)
        component_projection = torch.einsum("nki,ni->nk", component_grad, flat_vector)
        result = first + mean_grad * mean_projection[:, None]
        result = result - torch.einsum(
            "nk,nk,nki->ni", responsibilities, component_projection, component_grad
        )
        return result.reshape(*leading, self.dim)

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        indices = torch.multinomial(
            self.weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        noise = torch.randn(
            num_samples,
            self.dim,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        return self.means[indices] + torch.einsum("nij,nj->ni", self.cholesky[indices], noise)

    def sample_components(
        self, indices: torch.Tensor, scale: float, generator: torch.Generator
    ) -> torch.Tensor:
        noise = torch.randn(
            indices.shape[0],
            self.dim,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        return self.means[indices] + math.sqrt(scale) * torch.einsum(
            "nij,nj->ni", self.cholesky[indices], noise
        )

    def mode(self, x: torch.Tensor) -> torch.Tensor:
        return self.responsibilities(x).argmax(dim=-1)

    def true_moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.einsum("k,kd->d", self.weights, self.means)
        centered = self.means - mean
        covariance = torch.einsum("k,kij->ij", self.weights, self.covariances)
        covariance = covariance + torch.einsum("k,ki,kj->ij", self.weights, centered, centered)
        return mean, covariance

    def as_dict(self) -> dict[str, Any]:
        mean, covariance = self.true_moments()
        return {
            "weights": self.weights.detach().cpu().tolist(),
            "means": self.means.detach().cpu().tolist(),
            "covariances": self.covariances.detach().cpu().tolist(),
            "true_mean": mean.detach().cpu().tolist(),
            "true_covariance": covariance.detach().cpu().tolist(),
        }


def _diagonal_covariance(
    dim: int,
    condition_number: float,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if condition_number < 1.0:
        raise ValueError("条件数必须不小于 1")
    if dim == 1:
        eigenvalues = torch.tensor([scale], dtype=dtype, device=device)
    else:
        endpoint = -math.log10(condition_number)
        eigenvalues = scale * torch.logspace(0.0, endpoint, dim, dtype=dtype, device=device)
    return torch.diag(eigenvalues)


def build_target(
    config: TargetConfig,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> GaussianMixtureTarget:
    del seed
    device = torch.device(device)
    dim = config.dim
    family = config.family.lower()

    if family == "single":
        weights = torch.ones(1, dtype=dtype, device=device)
        means = torch.zeros(1, dim, dtype=dtype, device=device)
        covariance = _diagonal_covariance(
            dim, config.condition_number, config.covariance_scale, dtype, device
        )
        covariances = covariance.unsqueeze(0)

    elif family in {"bimodal", "asymmetric_bimodal"}:
        if dim < 2:
            raise ValueError("双模态目标至少需要二维")
        means = torch.zeros(2, dim, dtype=dtype, device=device)
        means[0, 0] = -config.separation
        means[1, 0] = config.separation
        diagonal = torch.ones(dim, dtype=dtype, device=device) * config.covariance_scale
        diagonal[1] = 0.25 * config.covariance_scale
        covariance = torch.diag(diagonal)
        covariances = covariance.unsqueeze(0).repeat(2, 1, 1)
        if family == "asymmetric_bimodal":
            weights = torch.tensor(
                [1.0 - config.rare_weight, config.rare_weight], dtype=dtype, device=device
            )
        else:
            weights = torch.full((2,), 0.5, dtype=dtype, device=device)

    elif family == "ring":
        if dim != 2:
            raise ValueError("环形混合目标固定为二维")
        modes = config.modes
        if modes < 3:
            raise ValueError("环形混合至少需要三个模态")
        component_std = 0.5 * math.sqrt(config.covariance_scale)
        radius = component_std * config.ring_mahalanobis / (2.0 * math.sin(math.pi / modes))
        angles = 2.0 * math.pi * torch.arange(modes, dtype=dtype, device=device) / modes
        means = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        covariances = (
            (0.25 * config.covariance_scale * torch.eye(2, dtype=dtype, device=device))
            .unsqueeze(0)
            .repeat(modes, 1, 1)
        )
        weights = torch.full((modes,), 1.0 / modes, dtype=dtype, device=device)

    elif family == "grid":
        if dim != 2:
            raise ValueError("网格混合目标固定为二维")
        grid_size = config.grid_size
        axis = (
            torch.arange(grid_size, dtype=dtype, device=device) - (grid_size - 1.0) / 2.0
        ) * config.grid_spacing
        mesh_x, mesh_y = torch.meshgrid(axis, axis, indexing="ij")
        means = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1)
        modes = means.shape[0]
        covariances = (
            (0.25 * config.covariance_scale * torch.eye(2, dtype=dtype, device=device))
            .unsqueeze(0)
            .repeat(modes, 1, 1)
        )
        weights = torch.full((modes,), 1.0 / modes, dtype=dtype, device=device)

    elif family == "highdim":
        if dim < 2:
            raise ValueError("高维混合目标至少需要二维")
        modes = config.modes
        covariance = _diagonal_covariance(
            dim, config.condition_number, config.covariance_scale, dtype, device
        )
        chol = torch.linalg.cholesky(covariance)
        angles = 2.0 * math.pi * torch.arange(modes, dtype=dtype, device=device) / modes
        whitened_radius = config.highdim_mahalanobis / (2.0 * math.sin(math.pi / modes))
        whitened = torch.zeros(modes, dim, dtype=dtype, device=device)
        whitened[:, 0] = whitened_radius * torch.cos(angles)
        whitened[:, 1] = whitened_radius * torch.sin(angles)
        means = torch.einsum("ij,kj->ki", chol, whitened)
        covariances = covariance.unsqueeze(0).repeat(modes, 1, 1)
        inverse_rank = 1.0 / torch.arange(1, modes + 1, dtype=dtype, device=device)
        weights = inverse_rank / inverse_rank.sum()

    else:
        raise ValueError(f"未知目标分布 family={config.family}")

    return GaussianMixtureTarget(weights=weights, means=means, covariances=covariances)
