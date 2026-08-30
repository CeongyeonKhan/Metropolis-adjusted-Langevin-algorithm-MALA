from __future__ import annotations

import math

import torch

from .config import GlobalProposalConfig
from .distributions import GaussianMixtureTarget


class GlobalMixtureProposal:
    """高斯—学生 t 独立混合提议；学生 t 参数使用尺度矩阵而非协方差矩阵。"""

    def __init__(self, reference: GaussianMixtureTarget, config: GlobalProposalConfig) -> None:
        self.reference = reference
        self.config = config

    def _gaussian_component_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, None, :] - self.reference.means[None, :, :]
        mahal = torch.einsum("nki,kij,nkj->nk", diff, self.reference.precisions, diff)
        scale = self.config.gaussian_scale
        normalizer = (
            self.reference.dim * math.log(2.0 * math.pi)
            + self.reference.logdet
            + self.reference.dim * math.log(scale)
        )
        return -0.5 * (normalizer[None, :] + mahal / scale)

    def _student_component_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, None, :] - self.reference.means[None, :, :]
        mahal = torch.einsum("nki,kij,nkj->nk", diff, self.reference.precisions, diff)
        scale = self.config.student_scale
        nu = self.config.degrees_of_freedom
        dim = self.reference.dim
        constant = (
            torch.lgamma(torch.tensor((nu + dim) / 2.0, dtype=x.dtype, device=x.device))
            - torch.lgamma(torch.tensor(nu / 2.0, dtype=x.dtype, device=x.device))
            - 0.5 * (self.reference.logdet + dim * math.log(scale) + dim * math.log(nu * math.pi))
        )
        return constant[None, :] - 0.5 * (nu + dim) * torch.log1p(mahal / (nu * scale))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_weights = self.reference.log_weights[None, :]
        gaussian = torch.logsumexp(log_weights + self._gaussian_component_log_prob(x), dim=-1)
        omega = self.config.student_weight
        if omega <= 0.0:
            return gaussian
        student = torch.logsumexp(log_weights + self._student_component_log_prob(x), dim=-1)
        return torch.logaddexp(
            gaussian + math.log1p(-omega),
            student + math.log(omega),
        )

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        components = torch.multinomial(
            self.reference.weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        normal = torch.randn(
            num_samples,
            self.reference.dim,
            dtype=self.reference.dtype,
            device=self.reference.device,
            generator=generator,
        )
        gaussian_part = torch.einsum("nij,nj->ni", self.reference.cholesky[components], normal)
        heavy = (
            torch.rand(
                num_samples,
                dtype=self.reference.dtype,
                device=self.reference.device,
                generator=generator,
            )
            < self.config.student_weight
        )
        output_scale = torch.full(
            (num_samples,),
            math.sqrt(self.config.gaussian_scale),
            dtype=self.reference.dtype,
            device=self.reference.device,
        )
        if bool(heavy.any()):
            nu = self.config.degrees_of_freedom
            alpha = torch.full(
                (num_samples,),
                nu / 2.0,
                dtype=self.reference.dtype,
                device=self.reference.device,
            )
            chi_square = 2.0 * torch._standard_gamma(alpha, generator=generator)
            t_scale = math.sqrt(self.config.student_scale) * torch.sqrt(nu / chi_square)
            output_scale = torch.where(heavy, t_scale, output_scale)
        return self.reference.means[components] + output_scale[:, None] * gaussian_part
