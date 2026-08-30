from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .adaptation import CONSTRAINT_NAMES, StepController
from .diagnostics import raw_frechet_distance
from .distributions import GaussianMixtureTarget
from .methods import MethodSpec


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_diagnostic_curves(metrics: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = metrics["axis/ceq_per_chain"]
    panels = [
        ("diagnostics/rhat_max", r"$\max\,\widehat{R}$"),
        ("diagnostics/ess_bulk_min", r"$\min\,ESS_{bulk}$"),
        ("distribution/fid_raw", r"$FID_{raw}$"),
        ("distribution/js", r"$D_{JS}$"),
    ]
    for axis, (column, title) in zip(axes.flat, panels):
        if column in metrics:
            values = pd.to_numeric(metrics[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            axis.plot(x, values, marker="o", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel(r"$C_{eq}$")
        axis.grid(alpha=0.25)
    return _save(fig, path)


def plot_trajectories(
    trajectory: torch.Tensor,
    target: GaussianMixtureTarget,
    path: Path,
) -> Path | None:
    if target.dim < 2 or trajectory.numel() == 0:
        return None
    points = trajectory[..., :2].detach().cpu().numpy()
    fig, axis = plt.subplots(figsize=(7, 6))
    for chain in range(points.shape[1]):
        axis.plot(points[:, chain, 0], points[:, chain, 1], alpha=0.75, linewidth=1.0)
        axis.scatter(points[0, chain, 0], points[0, chain, 1], marker="x", s=25)
    means = target.means[:, :2].detach().cpu().numpy()
    axis.scatter(means[:, 0], means[:, 1], c="black", marker="*", s=80, label=r"$\mu_k$")
    axis.set_title("Trajectories")
    axis.set_xlabel(r"$x_1$")
    axis.set_ylabel(r"$x_2$")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(fig, path)


def plot_mechanism_maps(
    target: GaussianMixtureTarget,
    controller: StepController,
    method: MethodSpec,
    grid_size: int,
    path: Path,
) -> Path | None:
    if target.dim < 2:
        return None
    means = target.means[:, :2]
    marginal_std = (
        torch.sqrt(torch.diagonal(target.covariances[:, :2, :2], dim1=-2, dim2=-1))
        .max(dim=0)
        .values
    )
    lower = means.min(dim=0).values - 3.0 * marginal_std
    upper = means.max(dim=0).values + 3.0 * marginal_std
    axis_x = torch.linspace(lower[0], upper[0], grid_size, dtype=target.dtype, device=target.device)
    axis_y = torch.linspace(lower[1], upper[1], grid_size, dtype=target.dtype, device=target.device)
    mesh_x, mesh_y = torch.meshgrid(axis_x, axis_y, indexing="xy")
    true_mean, _ = target.true_moments()
    points = true_mean[None, :].repeat(grid_size * grid_size, 1)
    points[:, 0] = mesh_x.reshape(-1)
    points[:, 1] = mesh_y.reshape(-1)
    with torch.no_grad():
        energy = target.energy(points)
        state = controller.evaluate(points, energy, method)
    fields = [
        (energy, r"$E_{acc}$"),
        (torch.sqrt(state.true_error_squared), r"$\delta^*$"),
        (torch.sqrt(state.proxy_error_squared), r"$\delta_U$"),
        (state.h, r"$h$"),
        (state.tau, r"$\tau$"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), constrained_layout=True)
    for subplot, (values, title) in zip(axes, fields):
        image = subplot.pcolormesh(
            mesh_x.detach().cpu().numpy(),
            mesh_y.detach().cpu().numpy(),
            values.reshape(grid_size, grid_size).detach().cpu().numpy(),
            shading="auto",
        )
        subplot.scatter(
            means[:, 0].detach().cpu(), means[:, 1].detach().cpu(), c="white", marker="x", s=25
        )
        subplot.set_title(title)
        fig.colorbar(image, ax=subplot, shrink=0.8)
    return _save(fig, path)


def plot_constraint_activation(activation: dict[str, float], path: Path) -> Path:
    labels = list(CONSTRAINT_NAMES) + ["numeric_floor"]
    values = [activation.get(label, 0.0) for label in labels]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values)
    axis.set_ylim(0.0, max(1.0, max(values, default=0.0) * 1.1))
    axis.set_ylabel(r"$p_{active}$")
    axis.set_title("Constraint activation")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def plot_adaptation_scatter(rows: np.ndarray, path: Path) -> Path | None:
    if rows.size == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(rows[:, 0], rows[:, 2], s=5, alpha=0.25)
    axes[0].set_xlabel(r"$E_{acc}$")
    axes[0].set_ylabel(r"$h$")
    axes[0].set_title(r"$E_{acc}$--$h$")
    axes[1].scatter(rows[:, 1], rows[:, 2], s=5, alpha=0.25)
    axes[1].set_xlabel(r"$\delta^*$")
    axes[1].set_ylabel(r"$h$")
    axes[1].set_title(r"$\delta^*$--$h$")
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(fig, path)


def plot_transition_matrix(matrix: torch.Tensor, path: Path) -> Path:
    values = matrix.detach().cpu().numpy()
    row_sum = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        values, row_sum, out=np.zeros_like(values, dtype=float), where=row_sum > 0
    )
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(probabilities, vmin=0.0, vmax=max(1.0e-12, probabilities.max()))
    axis.set_xlabel(r"$k_{t+1}$")
    axis.set_ylabel(r"$k_t$")
    axis.set_title(r"$P(k_{t+1}\mid k_t)$")
    fig.colorbar(image, ax=axis)
    return _save(fig, path)


def plot_survival(survival: dict[str, Any], path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.step(survival["time"], survival["survival"], where="post")
    axis.set_xlabel(r"$C_{eq}$")
    axis.set_ylabel(r"$\widehat{S}(C_{eq})$")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Kaplan--Meier")
    axis.grid(alpha=0.25)
    return _save(fig, path)


def plot_steady_state_bias(
    results: dict[str, torch.Tensor],
    target: GaussianMixtureTarget,
    path: Path,
) -> Path | None:
    if "A4" not in results or "A4-NC" not in results:
        return None
    true_mean, true_covariance = target.true_moments()
    labels, mean_error, covariance_error, fid = [], [], [], []
    for method in ("A4", "A4-NC"):
        samples = results[method].to(target.device)
        sample_mean = samples.mean(dim=0)
        centered = samples - sample_mean
        sample_cov = centered.transpose(0, 1) @ centered / max(1, samples.shape[0] - 1)
        labels.append(method)
        mean_error.append(float(torch.linalg.vector_norm(sample_mean - true_mean).cpu()))
        covariance_error.append(float(torch.linalg.matrix_norm(sample_cov - true_covariance).cpu()))
        fid.append(float(raw_frechet_distance(samples, true_mean, true_covariance).cpu()))
    x = np.arange(2)
    width = 0.24
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x - width, mean_error, width, label=r"$\|\Delta\mu\|_2$")
    axis.bar(x, covariance_error, width, label=r"$\|\Delta\Sigma\|_F$")
    axis.bar(x + width, fid, width, label=r"$FID_{raw}$")
    axis.set_xticks(x, labels)
    axis.set_title("A4 vs. A4-NC")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save(fig, path)
