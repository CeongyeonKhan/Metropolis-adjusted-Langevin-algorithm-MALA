from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .adaptation import StepController
from .calibration import BasinNormalizer, ErrorCalibrator, generate_calibration_points
from .config import ExperimentConfig, save_config
from .diagnostics import fid_reference_band
from .distributions import build_target
from .error_fields import GradientEnsemble
from .global_proposal import GlobalMixtureProposal
from .methods import METHOD_REGISTRY, get_method_spec
from .plotting import (
    plot_adaptation_scatter,
    plot_constraint_activation,
    plot_diagnostic_curves,
    plot_mechanism_maps,
    plot_steady_state_bias,
    plot_survival,
    plot_trajectories,
    plot_transition_matrix,
)
from .reference import fit_reference_mixture
from .sampler import SamplerRunResult, initialize_chains, run_sampler
from .tracking import ExperimentTracker
from .utils import (
    environment_metadata,
    make_generator,
    resolve_dtype,
    seed_everything,
    write_json,
)
from .whitening import AffineWhitening


def _resolve_output_root(config: ExperimentConfig) -> Path:
    root = Path(config.output.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root / config.output.experiment_name / f"seed_{config.seed}"


def _prepare_frozen_objects(
    config: ExperimentConfig,
    output_root: Path,
) -> tuple[
    Any,
    AffineWhitening,
    Any,
    GradientEnsemble,
    BasinNormalizer,
    ErrorCalibrator,
    GlobalMixtureProposal,
    dict[str, Any],
]:
    dtype = resolve_dtype(config.dtype)
    device = torch.device(config.device)
    original_target = build_target(config.target, dtype=dtype, device=device, seed=config.seed)
    whitening = AffineWhitening.fit(original_target, config.whitening)
    target = whitening.transform_target(original_target)
    reference, reference_metadata = fit_reference_mixture(
        target,
        config.reference,
        make_generator(config.seed, "reference_fit", device),
    )
    scale_points = generate_calibration_points(
        target,
        config.calibration,
        make_generator(config.seed, "error_scale_points", device),
    )
    normalizer_points = generate_calibration_points(
        target,
        config.calibration,
        make_generator(config.seed, "normalizer_points", device),
    )
    calibration_points = generate_calibration_points(
        target,
        config.calibration,
        make_generator(config.seed, "error_calibration_points", device),
    )
    ensemble_generator = make_generator(
        config.seed + config.error.seed_offset, "error_ensemble", device
    )
    ensemble = GradientEnsemble.create(
        target,
        config.error,
        scale_points,
        ensemble_generator,
    )
    normalizer = BasinNormalizer.fit(
        reference,
        ensemble,
        normalizer_points,
        z_max=config.adaptation.z_max,
        epsilon=config.adaptation.epsilon,
    )
    with torch.no_grad():
        members = ensemble.members(calibration_points)
        disagreement = ensemble.disagreement_variance(calibration_points, members)
        mean_gradient = members.mean(dim=1)
        true_error_squared = ensemble.true_error_squared(calibration_points, mean_gradient)
    calibrator, calibration_metadata = ErrorCalibrator.fit(
        disagreement,
        true_error_squared,
        config.calibration,
        epsilon=config.adaptation.epsilon,
    )
    test_start = calibration_metadata["fit_size"] + calibration_metadata["calibration_size"]
    test_points = calibration_points[test_start:]
    with torch.no_grad():
        test_disagreement = disagreement[test_start:]
        test_true = true_error_squared[test_start:]
        test_energy = target.energy(test_points)
        test_z_energy = normalizer.z_energy(test_points, test_energy)
        test_responsibilities = target.responsibilities(test_points)
        evaluation = calibrator.evaluate(
            test_disagreement,
            test_true,
            test_responsibilities,
            test_z_energy,
        )
    calibration_metadata["evaluation"] = evaluation
    global_proposal = GlobalMixtureProposal(reference, config.global_proposal)

    write_json(
        output_root / "target.json",
        {
            "original_coordinates": original_target.as_dict(),
            "working_coordinates": target.as_dict(),
            "whitening": whitening.as_dict(),
        },
    )
    write_json(output_root / "error_field.json", ensemble.as_dict())
    write_json(
        output_root / "reference_mixture.json",
        {"parameters": reference.as_dict(), "fit": reference_metadata},
    )
    write_json(output_root / "normalizer.json", normalizer.as_dict())
    write_json(output_root / "calibration.json", calibration_metadata)
    torch.save(
        {
            "amplitudes": ensemble.amplitudes.detach().cpu(),
            "frequencies": ensemble.frequencies.detach().cpu(),
            "phases": ensemble.phases.detach().cpu(),
            "scale": ensemble.scale.detach().cpu(),
            "calibrator": {
                "slope": calibrator.slope.detach().cpu(),
                "intercept": calibrator.intercept.detach().cpu(),
                "inflation": calibrator.inflation.detach().cpu(),
            },
        },
        output_root / "frozen_adaptation.pt",
    )
    return (
        original_target,
        whitening,
        target,
        ensemble,
        normalizer,
        calibrator,
        global_proposal,
        calibration_metadata,
    )


def _save_method_result(
    result: SamplerRunResult,
    run_dir: Path,
    config: ExperimentConfig,
    target: Any,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    metrics = pd.DataFrame(result.metrics)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "summary.json", result.summary)
    payload: dict[str, Any] = {
        "method": result.method,
        "final_samples": result.final_samples,
        "final_samples_raw": result.final_samples_raw,
        "trajectory": result.trajectory,
        "trajectory_raw": result.trajectory_raw,
        "trajectory_steps": result.trajectory_steps,
        "trajectory_cost_per_chain": result.trajectory_cost_per_chain,
        "transition_matrix": result.transition_matrix,
        "survival": result.survival,
    }
    if config.output.save_history:
        payload["history"] = result.history
        payload["history_steps"] = result.history_steps
        payload["history_cost_per_chain"] = result.history_cost_per_chain
    torch.save(payload, run_dir / "samples.pt")

    plot_dir = run_dir / "plots"
    images: dict[str, Path] = {}
    images["plots/diagnostic_curves"] = plot_diagnostic_curves(
        metrics, plot_dir / "diagnostic_curves.png"
    )
    trajectory_path = plot_trajectories(
        result.trajectory,
        target,
        plot_dir / "trajectories.png",
    )
    if trajectory_path is not None:
        images["plots/trajectories"] = trajectory_path
    images["plots/constraint_activation"] = plot_constraint_activation(
        result.activation, plot_dir / "constraint_activation.png"
    )
    scatter_path = plot_adaptation_scatter(result.scatter_rows, plot_dir / "adaptation_scatter.png")
    if scatter_path is not None:
        images["plots/adaptation_scatter"] = scatter_path
    images["plots/transition_matrix"] = plot_transition_matrix(
        result.transition_matrix, plot_dir / "transition_matrix.png"
    )
    images["plots/crossing_survival"] = plot_survival(
        result.survival, plot_dir / "crossing_survival.png"
    )
    return metrics, images


def _auc_to_budget(result: SamplerRunResult, metric: str, budget: float) -> float:
    x = np.asarray([row["axis/ceq_per_chain"] for row in result.metrics], dtype=float)
    y = np.asarray([row[metric] for row in result.metrics], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or budget <= x[0]:
        return 0.0
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    inside = x < budget
    bounded_x = np.concatenate([x[inside], np.asarray([budget])])
    bounded_y = np.concatenate([y[inside], np.asarray([np.interp(budget, x, y)])])
    intervals = np.diff(bounded_x)
    return float(np.sum(0.5 * (bounded_y[1:] + bounded_y[:-1]) * intervals))


def run_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    seed_everything(config.seed, config.deterministic)
    output_root = _resolve_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root / "resolved_config.yaml")
    write_json(output_root / "environment.json", environment_metadata())
    write_json(
        output_root / "method_capabilities.json",
        {name: spec.as_dict() for name, spec in METHOD_REGISTRY.items()},
    )
    preparation_start = time.perf_counter()
    (
        original_target,
        whitening,
        target,
        ensemble,
        normalizer,
        calibrator,
        global_proposal,
        calibration_metadata,
    ) = _prepare_frozen_objects(config, output_root)
    preparation_time = time.perf_counter() - preparation_start
    controller = StepController(ensemble, normalizer, calibrator, config.adaptation)

    initial_generator = make_generator(config.seed, "shared_initial_states", target.device)
    initial_states = initialize_chains(target, config, initial_generator)
    torch.save(
        {
            "working": initial_states.detach().cpu(),
            "raw": whitening.from_white(initial_states).detach().cpu(),
        },
        output_root / "initial_states.pt",
    )
    reference_generator = make_generator(config.seed, "fid_reference_band", target.device)
    reference_band = fid_reference_band(
        original_target,
        config.sampler.num_chains,
        config.metrics.fid_reference_repeats,
        reference_generator,
    )
    write_json(output_root / "fid_reference_band.json", reference_band)

    results: dict[str, SamplerRunResult] = {}
    all_summaries: list[dict[str, Any]] = []
    for method_name in config.sampler.methods:
        method = get_method_spec(method_name)
        run_dir = output_root / method_name
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "resolved_config.yaml")
        run_name = f"{config.output.experiment_name}-{method_name}-seed{config.seed}"
        group = (
            f"{config.output.experiment_name}-{config.target.family}-"
            f"{config.error.kind}-{config.sampler.initialization}"
        )
        with ExperimentTracker(
            run_dir,
            config.wandb,
            config.as_dict(),
            run_name,
            group,
        ) as tracker:
            result = run_sampler(
                config,
                target,
                original_target,
                whitening,
                controller,
                global_proposal,
                method,
                initial_states,
                reference_band,
                tracker,
            )
            result.summary["preparation_wall_time_s"] = preparation_time
            result.summary["calibration_test_coverage"] = calibration_metadata["evaluation"][
                "coverage"
            ]
            coverage_penalty = 100.0 * max(
                0.0,
                config.calibration.coverage - float(result.summary["calibration_test_coverage"]),
            )
            health_penalty = 1000.0 * float(
                result.summary["health/nonfinite_energy"]
                + result.summary["health/nonfinite_gradient"]
            )
            result.summary["sweep/objective"] = (
                float(result.summary["auc_js"])
                / max(float(result.summary["final_cost_per_chain"]), 1.0e-12)
                + coverage_penalty
                + health_penalty
            )
            _, images = _save_method_result(result, run_dir, config, target)
            if config.wandb.log_plots:
                if target.dim == 2:
                    mechanism_path = plot_mechanism_maps(
                        target,
                        controller,
                        method,
                        config.output.plot_grid_size,
                        run_dir / "plots" / "mechanism_maps.png",
                    )
                    if mechanism_path is not None:
                        images["plots/mechanism_maps"] = mechanism_path
                tracker.log_images(images)
            tracker.set_summary(result.summary)
            artifact_files = [
                run_dir / "resolved_config.yaml",
                run_dir / "metrics.csv",
                run_dir / "summary.json",
                run_dir / "samples.pt",
                output_root / "target.json",
                output_root / "reference_mixture.json",
                output_root / "error_field.json",
                output_root / "normalizer.json",
                output_root / "calibration.json",
                output_root / "frozen_adaptation.pt",
                output_root / "initial_states.pt",
                output_root / "fid_reference_band.json",
                output_root / "environment.json",
                output_root / "method_capabilities.json",
                *images.values(),
            ]
            tracker.log_artifact(
                artifact_files,
                name=f"{config.output.experiment_name}-{method_name}-seed{config.seed}",
            )
        results[method_name] = result
        all_summaries.append(result.summary)

    common_budget = min(result.summary["final_cost_per_chain"] for result in results.values())
    for method_name, result in results.items():
        result.summary["comparison/common_budget_per_chain"] = common_budget
        result.summary["comparison/auc_js_common"] = _auc_to_budget(
            result, "distribution/js", common_budget
        )
        result.summary["comparison/auc_fid_raw_common"] = _auc_to_budget(
            result, "distribution/fid_raw", common_budget
        )
        result.summary["comparison/mean_js_common"] = result.summary[
            "comparison/auc_js_common"
        ] / max(common_budget, 1.0e-12)
        result.summary["comparison/mean_fid_raw_common"] = result.summary[
            "comparison/auc_fid_raw_common"
        ] / max(common_budget, 1.0e-12)
        write_json(output_root / method_name / "summary.json", result.summary)

    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(output_root / "method_summary.csv", index=False)
    write_json(output_root / "method_summary.json", all_summaries)
    comparison = plot_steady_state_bias(
        {
            name: result.final_samples_raw.to(original_target.device)
            for name, result in results.items()
        },
        original_target,
        output_root / "a4_vs_a4_nc.png",
    )
    write_json(
        output_root / "run_manifest.json",
        {
            "output_root": str(output_root),
            "methods": config.sampler.methods,
            "comparison_plot": str(comparison) if comparison else None,
            "preparation_wall_time_s": preparation_time,
            "note": "正式采样使用解析接受能量；全部适配对象在采样前冻结。",
        },
    )
    return output_root
