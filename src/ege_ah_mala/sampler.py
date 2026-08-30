from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .adaptation import CONSTRAINT_NAMES, ProposalState, StepController, isotropic_log_proposal
from .config import ExperimentConfig
from .costs import CostCounter
from .diagnostics import ModeCrossingTracker, checkpoint_diagnostics
from .distributions import GaussianMixtureTarget
from .global_proposal import GlobalMixtureProposal
from .methods import MethodSpec
from .tracking import ExperimentTracker
from .utils import make_generator
from .whitening import AffineWhitening


def initialize_chains(
    target: GaussianMixtureTarget,
    config: ExperimentConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    num_chains = config.sampler.num_chains
    initialization = config.sampler.initialization.lower()
    if initialization == "stationary":
        return target.sample(num_chains, generator)
    if initialization == "single_mode":
        indices = torch.zeros(num_chains, dtype=torch.long, device=target.device)
        return target.sample_components(indices, scale=1.0, generator=generator)
    if initialization == "overdispersed":
        indices = torch.arange(num_chains, device=target.device) % target.num_components
        permutation = torch.randperm(num_chains, device=target.device, generator=generator)
        indices = indices[permutation]
        return target.sample_components(
            indices,
            scale=config.sampler.initialization_scale,
            generator=generator,
        )
    raise ValueError(f"未知初始化方式 {config.sampler.initialization}")


def _slice_state(state: ProposalState, selection: torch.Tensor) -> ProposalState:
    return ProposalState(
        **{
            field.name: getattr(state, field.name)[selection]
            for field in dataclasses.fields(ProposalState)
        }
    )


class ProposalStateCache:
    def __init__(self, num_chains: int, device: torch.device) -> None:
        self.num_chains = num_chains
        self.valid = torch.zeros(num_chains, dtype=torch.bool, device=device)
        self.values: dict[str, torch.Tensor] = {}

    def assign(self, indices: torch.Tensor, state: ProposalState) -> None:
        if indices.numel() == 0:
            return
        for field in dataclasses.fields(ProposalState):
            value = getattr(state, field.name)
            if field.name not in self.values:
                self.values[field.name] = torch.empty(
                    (self.num_chains, *value.shape[1:]), dtype=value.dtype, device=value.device
                )
            self.values[field.name][indices] = value
        self.valid[indices] = True

    def invalidate(self, indices: torch.Tensor) -> None:
        self.valid[indices] = False

    def select(self, indices: torch.Tensor) -> ProposalState:
        if not bool(self.valid[indices].all()):
            raise RuntimeError("尝试读取无效的提议状态缓存")
        return ProposalState(
            **{
                field.name: self.values[field.name][indices]
                for field in dataclasses.fields(ProposalState)
            }
        )

    def get_or_compute(
        self,
        indices: torch.Tensor,
        x: torch.Tensor,
        energy: torch.Tensor,
        controller: StepController,
        method: MethodSpec,
        costs: CostCounter,
    ) -> ProposalState:
        missing = indices[~self.valid[indices]]
        if missing.numel() > 0:
            computed = controller.evaluate(x[missing], energy[missing], method)
            costs.add_controller(
                int(missing.numel()),
                controller.ensemble.ensemble_size,
                controller.curvature_evaluations_per_point(method),
                indices=missing,
            )
            self.assign(missing, computed)
        return self.select(indices)


class RowReservoir:
    def __init__(self, capacity: int, columns: int, seed: int) -> None:
        self.capacity = capacity
        self.columns = columns
        self.data = np.empty((capacity, columns), dtype=np.float64)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, rows: np.ndarray) -> None:
        values = np.asarray(rows, dtype=np.float64).reshape(-1, self.columns)
        start = 0
        if self.size < self.capacity:
            take = min(self.capacity - self.size, values.shape[0])
            self.data[self.size : self.size + take] = values[:take]
            self.size += take
            self.seen += take
            start = take
        remaining = values[start:]
        if remaining.size == 0:
            return
        positions = np.arange(
            self.seen + 1,
            self.seen + remaining.shape[0] + 1,
            dtype=np.float64,
        )
        slots = np.floor(self.rng.random(remaining.shape[0]) * positions).astype(np.int64)
        selected = np.flatnonzero(slots < self.capacity)
        for row_index in selected:
            self.data[slots[row_index]] = remaining[row_index]
        self.seen += remaining.shape[0]

    def values(self) -> np.ndarray:
        return self.data[: self.size].copy()

    def clear(self) -> None:
        self.size = 0
        self.seen = 0


class SamplingMonitor:
    COLUMNS = (
        "energy",
        "true_delta",
        "proxy_delta",
        "h",
        "tau",
        "effective_step",
        "drift_scale",
        "z_energy",
        "z_gradient",
        "grad_rms",
    )

    def __init__(self, capacity: int, seed: int) -> None:
        self.window_rows = RowReservoir(capacity, len(self.COLUMNS), seed)
        self.plot_rows = RowReservoir(capacity, 3, seed + 1)
        self.window_constraints = np.zeros(len(CONSTRAINT_NAMES), dtype=np.int64)
        self.total_constraints = np.zeros(len(CONSTRAINT_NAMES), dtype=np.int64)
        self.window_numeric = 0
        self.total_numeric = 0
        self.window_below_operational = 0
        self.total_below_operational = 0
        self.window_state_count = 0
        self.total_state_count = 0
        self.window_local_proposed = 0
        self.window_local_accepted = 0
        self.window_global_proposed = 0
        self.window_global_accepted = 0
        self.total_local_proposed = 0
        self.total_local_accepted = 0
        self.total_global_proposed = 0
        self.total_global_accepted = 0
        self.nonfinite_energy = 0
        self.nonfinite_gradient = 0
        self.auto_reject = 0

    def add_state(self, state: ProposalState) -> None:
        rows = (
            torch.stack(
                [
                    state.energy,
                    torch.sqrt(state.true_error_squared.clamp_min(0.0)),
                    torch.sqrt(state.proxy_error_squared.clamp_min(0.0)),
                    state.h,
                    state.tau,
                    state.effective_step,
                    state.drift_scale,
                    state.z_energy,
                    state.z_gradient,
                    torch.linalg.vector_norm(state.gradient, dim=-1)
                    / math.sqrt(state.gradient.shape[-1]),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        self.window_rows.add(rows)
        self.plot_rows.add(rows[:, [0, 1, 3]])
        counts = torch.bincount(
            state.active_constraint.detach().cpu(), minlength=len(CONSTRAINT_NAMES)
        ).numpy()
        self.window_constraints += counts
        self.total_constraints += counts
        numeric = int(state.numeric_floor.sum().detach().cpu())
        below = int(state.below_operational.sum().detach().cpu())
        self.window_numeric += numeric
        self.total_numeric += numeric
        self.window_below_operational += below
        self.total_below_operational += below
        self.window_state_count += state.x.shape[0]
        self.total_state_count += state.x.shape[0]

    def add_acceptance(self, branch: str, proposed: int, accepted: int) -> None:
        if branch == "local":
            self.window_local_proposed += proposed
            self.window_local_accepted += accepted
            self.total_local_proposed += proposed
            self.total_local_accepted += accepted
        else:
            self.window_global_proposed += proposed
            self.window_global_accepted += accepted
            self.total_global_proposed += proposed
            self.total_global_accepted += accepted

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    def checkpoint_metrics(self) -> dict[str, float | int]:
        rows = self.window_rows.values()
        output: dict[str, float | int] = {}
        if rows.size:
            index = {name: position for position, name in enumerate(self.COLUMNS)}
            for name in (
                "h",
                "tau",
                "effective_step",
                "drift_scale",
                "energy",
                "true_delta",
                "proxy_delta",
            ):
                values = rows[:, index[name]]
                prefix = {
                    "h": "sampler/h",
                    "tau": "sampler/tau",
                    "effective_step": "sampler/effective_step",
                    "drift_scale": "sampler/drift_scale",
                    "energy": "state/energy",
                    "true_delta": "error/delta_true",
                    "proxy_delta": "error/delta_upper",
                }[name]
                output[f"{prefix}_mean"] = float(np.mean(values))
                output[f"{prefix}_p05"] = float(np.quantile(values, 0.05))
                output[f"{prefix}_median"] = float(np.quantile(values, 0.5))
                output[f"{prefix}_p95"] = float(np.quantile(values, 0.95))
            output["state/z_energy_mean"] = float(np.mean(rows[:, index["z_energy"]]))
            output["state/z_gradient_mean"] = float(np.mean(rows[:, index["z_gradient"]]))
            output["state/grad_rms_mean"] = float(np.mean(rows[:, index["grad_rms"]]))
            output["error/coverage"] = float(
                np.mean(rows[:, index["true_delta"]] <= rows[:, index["proxy_delta"]])
            )
            output["error/violation_rate"] = 1.0 - output["error/coverage"]
        else:
            for key in (
                "sampler/h_mean",
                "sampler/tau_mean",
                "sampler/effective_step_mean",
                "sampler/drift_scale_mean",
                "state/energy_mean",
                "error/delta_true_mean",
                "error/delta_upper_mean",
                "error/coverage",
            ):
                output[key] = float("nan")
        denominator = max(1, self.window_state_count)
        for index, name in enumerate(CONSTRAINT_NAMES):
            output[f"constraints/{name}"] = float(self.window_constraints[index] / denominator)
        output["constraints/numeric_floor"] = float(self.window_numeric / denominator)
        output["constraints/below_operational"] = float(self.window_below_operational / denominator)
        output["accept/local_window"] = self._rate(
            self.window_local_accepted, self.window_local_proposed
        )
        output["accept/global_window"] = self._rate(
            self.window_global_accepted, self.window_global_proposed
        )
        window_proposed = self.window_local_proposed + self.window_global_proposed
        window_accepted = self.window_local_accepted + self.window_global_accepted
        output["accept/overall_window"] = self._rate(window_accepted, window_proposed)
        output["accept/local_cumulative"] = self._rate(
            self.total_local_accepted, self.total_local_proposed
        )
        output["accept/global_cumulative"] = self._rate(
            self.total_global_accepted, self.total_global_proposed
        )
        total_proposed = self.total_local_proposed + self.total_global_proposed
        total_accepted = self.total_local_accepted + self.total_global_accepted
        output["accept/overall_cumulative"] = self._rate(total_accepted, total_proposed)
        output["sampler/global_kernel_fraction"] = self._rate(
            self.total_global_proposed, total_proposed
        )
        output["health/nonfinite_energy"] = self.nonfinite_energy
        output["health/nonfinite_gradient"] = self.nonfinite_gradient
        output["health/auto_reject"] = self.auto_reject
        self.window_rows.clear()
        self.window_constraints.fill(0)
        self.window_numeric = 0
        self.window_below_operational = 0
        self.window_state_count = 0
        self.window_local_proposed = 0
        self.window_local_accepted = 0
        self.window_global_proposed = 0
        self.window_global_accepted = 0
        return output

    def final_activation(self) -> dict[str, float]:
        denominator = max(1, self.total_state_count)
        output = {
            name: float(self.total_constraints[index] / denominator)
            for index, name in enumerate(CONSTRAINT_NAMES)
        }
        output["numeric_floor"] = float(self.total_numeric / denominator)
        output["below_operational"] = float(self.total_below_operational / denominator)
        return output


@dataclass
class SamplerRunResult:
    method: str
    metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    final_samples: torch.Tensor
    final_samples_raw: torch.Tensor
    history: torch.Tensor
    history_steps: torch.Tensor
    history_cost_per_chain: torch.Tensor
    trajectory: torch.Tensor
    trajectory_raw: torch.Tensor
    trajectory_steps: torch.Tensor
    trajectory_cost_per_chain: torch.Tensor
    transition_matrix: torch.Tensor
    survival: dict[str, Any]
    activation: dict[str, float]
    scatter_rows: np.ndarray


def run_sampler(
    config: ExperimentConfig,
    target: GaussianMixtureTarget,
    raw_target: GaussianMixtureTarget,
    whitening: AffineWhitening,
    controller: StepController,
    global_proposal: GlobalMixtureProposal,
    method: MethodSpec,
    initial_states: torch.Tensor,
    reference_band: dict[str, float],
    tracker: ExperimentTracker,
) -> SamplerRunResult:
    num_chains = config.sampler.num_chains
    device = target.device
    x = initial_states.clone()
    with torch.no_grad():
        energy = target.energy(x)
    if not bool(torch.isfinite(energy).all()):
        raise RuntimeError("初始状态存在非有限接受能量")
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    route_generator = make_generator(config.seed, "formal_route", device)
    local_generator = make_generator(config.seed, "formal_local_noise", device)
    global_generator = make_generator(config.seed, "formal_global_proposal", device)
    accept_generator = make_generator(config.seed, "formal_accept", device)
    costs = CostCounter(num_chains=num_chains, device=device)
    cache = ProposalStateCache(num_chains, device)
    monitor = SamplingMonitor(config.output.reservoir_size, config.seed + 71)
    history: list[torch.Tensor] = [x.detach().cpu().clone()]
    history_steps: list[int] = [0]
    history_cost_per_chain: list[float] = [0.0]
    recorded = min(num_chains, config.sampler.recorded_chains)
    trajectory: list[torch.Tensor] = [x[:recorded].detach().cpu().clone()]
    trajectory_steps: list[int] = [0]
    trajectory_cost_per_chain: list[float] = [0.0]
    initial_responsibilities = target.responsibilities(x)
    crossing = ModeCrossingTracker.create(
        initial_responsibilities,
        config.metrics.mode_confidence,
        config.metrics.mode_confirmation_checkpoints,
    )
    metrics_rows: list[dict[str, Any]] = []
    checkpoints = set(config.sampler.checkpoints)
    sampling_start = time.perf_counter()
    metrics_time = 0.0
    consecutive_converged = 0
    convergence_cost = float("nan")

    def record_checkpoint(step: int) -> None:
        nonlocal metrics_time, consecutive_converged, convergence_cost
        checkpoint_start = time.perf_counter()
        history_tensor = torch.stack(history, dim=0).to(device)
        ceq = costs.equivalent(config.cost)
        ceq_per_chain = ceq / num_chains
        chain_costs = costs.equivalent_by_chain(config.cost)
        row: dict[str, Any] = {
            "axis/transition_per_chain": step,
            "axis/ceq_per_chain": ceq_per_chain,
            "axis/wall_time_s": time.perf_counter() - sampling_start,
            "method": method.name,
        }
        row.update(monitor.checkpoint_metrics())
        row.update(
            checkpoint_diagnostics(
                history_tensor,
                x,
                target,
                config.metrics,
                reference_band,
                fid_current=whitening.from_white(x),
                fid_target=raw_target,
            )
        )
        row.update(crossing.summary(chain_costs.clamp_min(1.0e-12)))
        row["cost/ceq_per_chain_min"] = float(chain_costs.min().cpu())
        row["cost/ceq_per_chain_max"] = float(chain_costs.max().cpu())
        cost_values = costs.as_dict(config.cost, num_chains)
        row.update(
            {
                "cost/n_energy": cost_values["n_energy"],
                "cost/n_gradient": cost_values["n_gradient"],
                "cost/n_hessian": cost_values["n_hessian"],
                "cost/n_global_density": cost_values["n_global_density"],
                "cost/n_diffusion": cost_values["n_diffusion"],
                "cost/ceq_total": cost_values["equivalent_total"],
                "cost/ceq_per_chain": cost_values["equivalent_per_chain"],
            }
        )
        rhat = row["diagnostics/rhat_max"]
        ess_bulk = row["diagnostics/ess_bulk_min"]
        ess_tail = row["diagnostics/ess_tail_min"]
        global_convergence_applicable = (
            target.num_components == 1
            or config.sampler.initialization.lower() in {"stationary", "overdispersed"}
        )
        global_coverage_ok = (
            float(row["distribution/js"]) <= config.metrics.max_js_for_convergence
            and float(row["distribution/min_mode_ratio"])
            >= config.metrics.min_mode_ratio_for_convergence
        )
        meets = (
            global_convergence_applicable
            and global_coverage_ok
            and isinstance(rhat, (float, int))
            and math.isfinite(float(rhat))
            and float(rhat) < 1.01
            and math.isfinite(float(ess_bulk))
            and float(ess_bulk) >= 400.0
            and math.isfinite(float(ess_tail))
            and float(ess_tail) >= 100.0
        )
        row["diagnostics/global_convergence_applicable"] = int(global_convergence_applicable)
        row["diagnostics/global_coverage_ok"] = int(global_coverage_ok)
        consecutive_converged = consecutive_converged + 1 if meets else 0
        row["diagnostics/converged"] = int(consecutive_converged >= 5)
        if consecutive_converged >= 5 and not math.isfinite(convergence_cost):
            convergence_cost = ceq_per_chain
        metrics_time += time.perf_counter() - checkpoint_start
        row["runtime/metrics_s"] = metrics_time
        row["runtime/sampling_s"] = time.perf_counter() - sampling_start - metrics_time
        if torch.cuda.is_available() and device.type == "cuda":
            row["runtime/peak_cuda_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        else:
            row["runtime/peak_cuda_mb"] = 0.0
        metrics_rows.append(row)
        tracker.log(row)

    record_checkpoint(0)
    tiny = torch.finfo(target.dtype).tiny
    with torch.no_grad():
        for step in range(1, config.sampler.num_steps + 1):
            route = torch.rand(
                num_chains, dtype=target.dtype, device=device, generator=route_generator
            )
            local_noise = torch.randn(
                num_chains,
                target.dim,
                dtype=target.dtype,
                device=device,
                generator=local_generator,
            )
            accept_uniform = torch.rand(
                num_chains, dtype=target.dtype, device=device, generator=accept_generator
            ).clamp_min(tiny)
            if method.global_kernel:
                global_mask = route < config.global_proposal.probability
            else:
                global_mask = torch.zeros(num_chains, dtype=torch.bool, device=device)
            local_mask = ~global_mask

            local_indices = torch.where(local_mask)[0]
            if local_indices.numel() > 0:
                state_x = cache.get_or_compute(local_indices, x, energy, controller, method, costs)
                if not bool(state_x.finite.all()):
                    monitor.nonfinite_gradient += int((~state_x.finite).sum().cpu())
                    raise RuntimeError("当前状态的梯度或提议参数出现非有限值，正式链已终止")
                monitor.add_state(state_x)
                mean = (
                    state_x.x
                    - state_x.h[:, None] * state_x.drift_multiplier[:, None] * state_x.gradient
                )
                local_candidate = (
                    mean
                    + torch.sqrt(2.0 * state_x.h * state_x.tau)[:, None]
                    * local_noise[local_indices]
                )
                candidate_energy = target.energy(local_candidate)
                costs.add_energy(int(local_indices.numel()), indices=local_indices)
                finite_candidate = torch.isfinite(local_candidate).all(dim=-1) & torch.isfinite(
                    candidate_energy
                )
                monitor.nonfinite_energy += int((~finite_candidate).sum().cpu())
                accepted_local = torch.zeros(local_indices.numel(), dtype=torch.bool, device=device)
                if method.corrected:
                    valid_position = torch.where(finite_candidate)[0]
                    if valid_position.numel() > 0:
                        state_y = controller.evaluate(
                            local_candidate[valid_position],
                            candidate_energy[valid_position],
                            method,
                        )
                        costs.add_controller(
                            int(valid_position.numel()),
                            controller.ensemble.ensemble_size,
                            controller.curvature_evaluations_per_point(method),
                            indices=local_indices[valid_position],
                        )
                        valid_state_position = torch.where(state_y.finite)[0]
                        monitor.nonfinite_gradient += int((~state_y.finite).sum().cpu())
                        monitor.auto_reject += int((~state_y.finite).sum().cpu())
                        if valid_state_position.numel() > 0:
                            good_position = valid_position[valid_state_position]
                            good_state_y = _slice_state(state_y, valid_state_position)
                            good_state_x = _slice_state(state_x, good_position)
                            log_forward = isotropic_log_proposal(
                                local_candidate[good_position], good_state_x
                            )
                            log_reverse = isotropic_log_proposal(
                                x[local_indices[good_position]], good_state_y
                            )
                            log_ratio = (
                                -candidate_energy[good_position]
                                + energy[local_indices[good_position]]
                                + log_reverse
                                - log_forward
                            )
                            finite_ratio = torch.isfinite(log_ratio)
                            decision = torch.log(
                                accept_uniform[local_indices[good_position]]
                            ) < torch.minimum(torch.zeros_like(log_ratio), log_ratio)
                            decision &= finite_ratio
                            accepted_local[good_position] = decision
                            accepted_good_position = torch.where(decision)[0]
                            if accepted_good_position.numel() > 0:
                                cache.assign(
                                    local_indices[good_position[accepted_good_position]],
                                    _slice_state(good_state_y, accepted_good_position),
                                )
                            monitor.auto_reject += int((~finite_ratio).sum().cpu())
                    monitor.auto_reject += int((~finite_candidate).sum().cpu())
                else:
                    accepted_local = finite_candidate
                    cache.invalidate(local_indices[accepted_local])
                    monitor.auto_reject += int((~finite_candidate).sum().cpu())
                if bool(accepted_local.any()):
                    accepted_indices = local_indices[accepted_local]
                    x[accepted_indices] = local_candidate[accepted_local]
                    energy[accepted_indices] = candidate_energy[accepted_local]
                monitor.add_acceptance(
                    "local", int(local_indices.numel()), int(accepted_local.sum().cpu())
                )

            global_indices = torch.where(global_mask)[0]
            if global_indices.numel() > 0:
                global_candidate = global_proposal.sample(
                    int(global_indices.numel()), global_generator
                )
                candidate_energy = target.energy(global_candidate)
                costs.add_energy(int(global_indices.numel()), indices=global_indices)
                finite_candidate = torch.isfinite(global_candidate).all(dim=-1) & torch.isfinite(
                    candidate_energy
                )
                monitor.nonfinite_energy += int((~finite_candidate).sum().cpu())
                accepted_global = torch.zeros(
                    global_indices.numel(), dtype=torch.bool, device=device
                )
                if method.corrected:
                    valid_position = torch.where(finite_candidate)[0]
                    if valid_position.numel() > 0:
                        log_q_current = global_proposal.log_prob(x[global_indices[valid_position]])
                        log_q_candidate = global_proposal.log_prob(global_candidate[valid_position])
                        costs.add_global_density(
                            2 * int(valid_position.numel()),
                            indices=global_indices[valid_position],
                            evaluations_per_chain=2,
                        )
                        log_ratio = (
                            -candidate_energy[valid_position]
                            + energy[global_indices[valid_position]]
                            + log_q_current
                            - log_q_candidate
                        )
                        finite_ratio = torch.isfinite(log_ratio)
                        decision = torch.log(
                            accept_uniform[global_indices[valid_position]]
                        ) < torch.minimum(torch.zeros_like(log_ratio), log_ratio)
                        decision &= finite_ratio
                        accepted_global[valid_position] = decision
                        monitor.auto_reject += int((~finite_ratio).sum().cpu())
                    monitor.auto_reject += int((~finite_candidate).sum().cpu())
                else:
                    accepted_global = finite_candidate
                    monitor.auto_reject += int((~finite_candidate).sum().cpu())
                if bool(accepted_global.any()):
                    accepted_indices = global_indices[accepted_global]
                    x[accepted_indices] = global_candidate[accepted_global]
                    energy[accepted_indices] = candidate_energy[accepted_global]
                    cache.invalidate(accepted_indices)
                monitor.add_acceptance(
                    "global", int(global_indices.numel()), int(accepted_global.sum().cpu())
                )

            if step % config.sampler.crossing_stride == 0:
                crossing.update(
                    target.responsibilities(x),
                    costs.equivalent_by_chain(config.cost),
                )
            if step % config.sampler.diagnostic_stride == 0:
                history.append(x.detach().cpu().clone())
                history_steps.append(step)
                history_cost_per_chain.append(costs.equivalent(config.cost) / num_chains)
            if step % config.sampler.trajectory_stride == 0:
                trajectory.append(x[:recorded].detach().cpu().clone())
                trajectory_steps.append(step)
                trajectory_cost_per_chain.append(costs.equivalent(config.cost) / num_chains)
            if step in checkpoints:
                record_checkpoint(step)

    sampling_elapsed = time.perf_counter() - sampling_start
    final_cost = costs.equivalent(config.cost) / num_chains
    final_chain_costs = costs.equivalent_by_chain(config.cost)
    crossing_summary = crossing.summary(final_chain_costs.clamp_min(1.0e-12))
    final_row = metrics_rows[-1]
    x_axis = np.asarray([row["axis/ceq_per_chain"] for row in metrics_rows], dtype=float)
    js_values = np.asarray([row["distribution/js"] for row in metrics_rows], dtype=float)
    fid_values = np.asarray([row["distribution/fid_raw"] for row in metrics_rows], dtype=float)
    if x_axis.size > 1:
        interval = np.diff(x_axis)
        auc_js = float(np.sum(0.5 * (js_values[1:] + js_values[:-1]) * interval))
        auc_fid = float(np.sum(0.5 * (fid_values[1:] + fid_values[:-1]) * interval))
    else:
        auc_js = 0.0
        auc_fid = 0.0
    final_samples_raw = whitening.from_white(x)
    true_mean, true_covariance = raw_target.true_moments()
    sample_mean = final_samples_raw.mean(dim=0)
    centered = final_samples_raw - sample_mean
    sample_covariance = centered.transpose(0, 1) @ centered / max(1, num_chains - 1)
    summary: dict[str, Any] = {
        "method": method.name,
        "final_cost_per_chain": final_cost,
        "convergence_cost_per_chain": convergence_cost,
        "final_rhat_max": final_row["diagnostics/rhat_max"],
        "final_ess_bulk_min": final_row["diagnostics/ess_bulk_min"],
        "final_ess_tail_min": final_row["diagnostics/ess_tail_min"],
        "global_convergence_applicable": final_row["diagnostics/global_convergence_applicable"],
        "global_converged": final_row["diagnostics/converged"],
        "final_fid_raw": final_row["distribution/fid_raw"],
        "final_js": final_row["distribution/js"],
        "auc_js": auc_js,
        "auc_fid_raw": auc_fid,
        "ess_bulk_per_cost": (
            float(final_row["diagnostics/ess_bulk_min"]) / final_cost
            if final_cost > 0 and math.isfinite(float(final_row["diagnostics/ess_bulk_min"]))
            else float("nan")
        ),
        "mean_error_raw": float(torch.linalg.vector_norm(sample_mean - true_mean).cpu()),
        "covariance_error_raw": float(
            torch.linalg.matrix_norm(sample_covariance - true_covariance).cpu()
        ),
        "sampling_wall_time_s": sampling_elapsed,
        "metrics_wall_time_s": metrics_time,
        "local_acceptance": SamplingMonitor._rate(
            monitor.total_local_accepted, monitor.total_local_proposed
        ),
        "global_acceptance": SamplingMonitor._rate(
            monitor.total_global_accepted, monitor.total_global_proposed
        ),
        "overall_acceptance": SamplingMonitor._rate(
            monitor.total_local_accepted + monitor.total_global_accepted,
            monitor.total_local_proposed + monitor.total_global_proposed,
        ),
        "health/nonfinite_energy": monitor.nonfinite_energy,
        "health/nonfinite_gradient": monitor.nonfinite_gradient,
        "health/auto_reject": monitor.auto_reject,
    }
    summary.update(crossing_summary)
    summary.update(
        {f"activation/{key}": value for key, value in monitor.final_activation().items()}
    )
    tracker.set_summary(summary)
    return SamplerRunResult(
        method=method.name,
        metrics=metrics_rows,
        summary=summary,
        final_samples=x.detach().cpu(),
        final_samples_raw=final_samples_raw.detach().cpu(),
        history=torch.stack(history, dim=0),
        history_steps=torch.tensor(history_steps, dtype=torch.long),
        history_cost_per_chain=torch.tensor(history_cost_per_chain, dtype=torch.float64),
        trajectory=torch.stack(trajectory, dim=0),
        trajectory_raw=whitening.from_white(torch.stack(trajectory, dim=0).to(device))
        .detach()
        .cpu(),
        trajectory_steps=torch.tensor(trajectory_steps, dtype=torch.long),
        trajectory_cost_per_chain=torch.tensor(trajectory_cost_per_chain, dtype=torch.float64),
        transition_matrix=crossing.transition_matrix.detach().cpu(),
        survival=crossing.survival_data(final_chain_costs.clamp_min(1.0e-12)),
        activation=monitor.final_activation(),
        scatter_rows=monitor.plot_rows.values(),
    )
