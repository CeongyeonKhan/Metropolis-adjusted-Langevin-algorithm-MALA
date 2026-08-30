from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import WhiteningConfig
from .distributions import GaussianMixtureTarget


@dataclass
class AffineWhitening:
    offset: torch.Tensor
    cholesky: torch.Tensor
    enabled: bool
    source: str

    @classmethod
    def fit(
        cls,
        target: GaussianMixtureTarget,
        config: WhiteningConfig,
    ) -> AffineWhitening:
        offset, target_covariance = target.true_moments()
        identity = torch.eye(target.dim, dtype=target.dtype, device=target.device)
        if not config.enabled or config.source == "identity":
            return cls(
                offset=torch.zeros_like(offset), cholesky=identity, enabled=False, source="identity"
            )
        if config.source == "pooled_component":
            scale = torch.einsum("k,kij->ij", target.weights, target.covariances)
        elif config.source == "target_moments":
            scale = target_covariance
        else:
            raise ValueError(f"未知白化来源 {config.source}")
        scale = 0.5 * (scale + scale.transpose(-1, -2)) + config.jitter * identity
        return cls(
            offset=offset,
            cholesky=torch.linalg.cholesky(scale),
            enabled=True,
            source=config.source,
        )

    def to_white(self, original: torch.Tensor) -> torch.Tensor:
        centered = original - self.offset
        return torch.linalg.solve_triangular(
            self.cholesky,
            centered.transpose(-1, -2),
            upper=False,
        ).transpose(-1, -2)

    def from_white(self, whitened: torch.Tensor) -> torch.Tensor:
        return self.offset + torch.einsum("ij,...j->...i", self.cholesky, whitened)

    def transform_target(self, target: GaussianMixtureTarget) -> GaussianMixtureTarget:
        means = self.to_white(target.means)
        inverse = torch.linalg.inv(self.cholesky)
        covariances = torch.einsum(
            "ij,kjl,ml->kim",
            inverse,
            target.covariances,
            inverse,
        )
        return GaussianMixtureTarget(
            weights=target.weights.clone(),
            means=means,
            covariances=covariances,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "offset": self.offset.detach().cpu().tolist(),
            "cholesky": self.cholesky.detach().cpu().tolist(),
            "log_abs_det": float(torch.logdet(self.cholesky).detach().cpu()),
        }
