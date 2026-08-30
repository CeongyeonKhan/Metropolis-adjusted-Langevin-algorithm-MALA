from __future__ import annotations

import torch

from .config import ReferenceConfig
from .distributions import GaussianMixtureTarget


def _sample_covariance(points: torch.Tensor, jitter: float) -> torch.Tensor:
    centered = points - points.mean(dim=0)
    covariance = centered.transpose(0, 1) @ centered / max(1, points.shape[0] - 1)
    identity = torch.eye(points.shape[1], dtype=points.dtype, device=points.device)
    return 0.5 * (covariance + covariance.transpose(-1, -2)) + jitter * identity


def _farthest_point_initialization(points: torch.Tensor, components: int) -> torch.Tensor:
    center = points.mean(dim=0)
    first = torch.argmax((points - center).square().sum(dim=-1))
    selected = [points[first]]
    minimum_distance = (points - selected[0]).square().sum(dim=-1)
    for _ in range(1, components):
        index = torch.argmax(minimum_distance)
        selected.append(points[index])
        distance = (points - selected[-1]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    return torch.stack(selected)


def fit_reference_mixture(
    accepted_target: GaussianMixtureTarget,
    config: ReferenceConfig,
    generator: torch.Generator,
) -> tuple[GaussianMixtureTarget, dict[str, object]]:
    if config.source == "analytic":
        reference = GaussianMixtureTarget(
            weights=accepted_target.weights.clone(),
            means=accepted_target.means.clone(),
            covariances=accepted_target.covariances.clone(),
        )
        return reference, {"source": "analytic", "num_points": 0, "em_iterations": 0}

    points = accepted_target.sample(config.num_points, generator)
    components = accepted_target.num_components
    means = _farthest_point_initialization(points, components)
    common_covariance = _sample_covariance(points, config.covariance_jitter)
    covariances = common_covariance[None, :, :].repeat(components, 1, 1)
    weights = torch.full(
        (components,),
        1.0 / components,
        dtype=points.dtype,
        device=points.device,
    )
    identity = torch.eye(points.shape[1], dtype=points.dtype, device=points.device)
    for _ in range(config.em_iterations):
        mixture = GaussianMixtureTarget(weights, means, covariances)
        responsibilities = mixture.responsibilities(points)
        effective = responsibilities.sum(dim=0).clamp_min(1.0e-8)
        weights = (effective / effective.sum()).clamp_min(config.min_weight)
        weights = weights / weights.sum()
        means = torch.einsum("nk,nd->kd", responsibilities, points) / effective[:, None]
        difference = points[:, None, :] - means[None, :, :]
        covariances = (
            torch.einsum("nk,nki,nkj->kij", responsibilities, difference, difference)
            / effective[:, None, None]
        )
        covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
        covariances = covariances + config.covariance_jitter * identity[None, :, :]
    reference = GaussianMixtureTarget(weights, means, covariances)
    mean_log_likelihood = float(reference.log_prob(points).mean().detach().cpu())
    return reference, {
        "source": "independent_em_fit",
        "num_points": config.num_points,
        "em_iterations": config.em_iterations,
        "mean_log_likelihood": mean_log_likelihood,
        "covariance_jitter": config.covariance_jitter,
    }
