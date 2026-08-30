from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TargetConfig:
    family: str = "bimodal"
    dim: int = 2
    modes: int = 2
    separation: float = 4.0
    condition_number: float = 4.0
    covariance_scale: float = 1.0
    rare_weight: float = 0.1
    ring_mahalanobis: float = 4.0
    grid_size: int = 3
    grid_spacing: float = 2.0
    highdim_mahalanobis: float = 6.0


@dataclass
class WhiteningConfig:
    enabled: bool = True
    source: str = "pooled_component"
    jitter: float = 1.0e-8


@dataclass
class ReferenceConfig:
    source: str = "fitted"
    num_points: int = 4096
    em_iterations: int = 40
    covariance_jitter: float = 1.0e-4
    min_weight: float = 1.0e-4


@dataclass
class ErrorConfig:
    kind: str = "fourier"
    ensemble_size: int = 5
    features: int = 64
    relative_rmse: float = 0.2
    frequency_scale: float = 0.7
    scale_bias: float = 0.2
    boundary_gain: float = 3.0
    seed_offset: int = 101


@dataclass
class CalibrationConfig:
    num_points: int = 2048
    fit_fraction: float = 0.5
    calibration_fraction: float = 0.25
    coverage: float = 0.95
    wide_scale: float = 3.0


@dataclass
class AdaptationConfig:
    h0: float = 0.05
    h_max: float = 0.5
    h_num: float = 1.0e-8
    h_oper: float = 1.0e-5
    z_max: float = 3.0
    alpha_energy: float = 0.25
    beta_energy: float = 2.0
    beta_gradient: float = 2.0
    tau_max: float = 2.0
    gamma: float = 1.0
    curvature_c: float = 0.5
    curvature_safety: float = 1.5
    curvature_mode: str = "exact"
    curvature_probes: int = 2
    curvature_power_iterations: int = 4
    epsilon_proposal: float = 0.05
    drift_radius_factor: float = 0.5
    epsilon: float = 1.0e-12


@dataclass
class GlobalProposalConfig:
    probability: float = 0.05
    gaussian_scale: float = 2.0
    student_scale: float = 4.0
    student_weight: float = 0.05
    degrees_of_freedom: float = 5.0


@dataclass
class SamplerConfig:
    methods: list[str] = field(default_factory=lambda: ["B1", "B2", "A2-P", "A4"])
    num_chains: int = 64
    num_steps: int = 5000
    initialization: str = "single_mode"
    initialization_scale: float = 4.0
    diagnostic_stride: int = 1
    trajectory_stride: int = 10
    crossing_stride: int = 10
    recorded_chains: int = 8
    checkpoints: list[int] = field(
        default_factory=lambda: [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    )


@dataclass
class MetricsConfig:
    fid_reference_repeats: int = 100
    mode_confidence: float = 0.9
    mode_confirmation_checkpoints: int = 3
    min_rhat_draws: int = 8
    max_js_for_convergence: float = 0.05
    min_mode_ratio_for_convergence: float = 0.5


@dataclass
class CostConfig:
    energy_weight: float = 0.1
    hessian_weight: float = 2.0
    global_density_weight: float = 0.1
    diffusion_weight: float = 1.0


@dataclass
class WandbConfig:
    mode: str = "offline"
    project: str = "ege-ah-mala"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=lambda: ["synthetic", "mala", "pytorch"])
    log_plots: bool = True
    log_artifact: bool = True
    strict: bool = False


@dataclass
class OutputConfig:
    root: str = "outputs"
    experiment_name: str = "p5_full_bimodal"
    save_history: bool = True
    plot_grid_size: int = 80
    reservoir_size: int = 10000


@dataclass
class ExperimentConfig:
    seed: int = 20260829
    device: str = "cpu"
    dtype: str = "float64"
    deterministic: bool = True
    target: TargetConfig = field(default_factory=TargetConfig)
    whitening: WhiteningConfig = field(default_factory=WhiteningConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    error: ErrorConfig = field(default_factory=ErrorConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    global_proposal: GlobalProposalConfig = field(default_factory=GlobalProposalConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype 仅支持 float32 或 float64")
        if self.target.dim < 1:
            raise ValueError("target.dim 必须为正整数")
        if self.target.covariance_scale <= 0 or self.target.condition_number < 1:
            raise ValueError("目标协方差尺度必须为正，条件数不得小于 1")
        valid_families = {
            "single",
            "bimodal",
            "asymmetric_bimodal",
            "ring",
            "grid",
            "highdim",
        }
        if self.target.family not in valid_families:
            raise ValueError(f"未知目标分布 family={self.target.family}")
        if self.target.family == "ring" and self.target.modes < 3:
            raise ValueError("环形混合至少需要三个模态")
        if self.target.family == "highdim" and self.target.modes < 2:
            raise ValueError("高维混合至少需要两个模态")
        if self.target.family == "grid" and self.target.grid_size < 2:
            raise ValueError("网格混合每轴至少需要两个模态")
        if not 0.0 < self.target.rare_weight < 1.0:
            raise ValueError("稀有模态权重必须位于 (0, 1)")
        if self.whitening.source not in {"pooled_component", "target_moments", "identity"}:
            raise ValueError("whitening.source 必须为 pooled_component、target_moments 或 identity")
        if self.whitening.jitter <= 0:
            raise ValueError("whitening.jitter 必须为正")
        if self.reference.source not in {"fitted", "analytic"}:
            raise ValueError("reference.source 必须为 fitted 或 analytic")
        if self.target.family == "single":
            expected_components = 1
        elif self.target.family in {"bimodal", "asymmetric_bimodal"}:
            expected_components = 2
        elif self.target.family == "grid":
            expected_components = self.target.grid_size**2
        else:
            expected_components = self.target.modes
        if self.reference.num_points < expected_components:
            raise ValueError("reference.num_points 不得小于目标模态数")
        if self.reference.em_iterations < 1 or self.reference.covariance_jitter <= 0:
            raise ValueError("参考混合的迭代次数和协方差扰动量必须为正")
        if not 0.0 < self.reference.min_weight < 1.0:
            raise ValueError("参考混合最小权重必须位于 (0, 1)")
        if self.reference.min_weight * expected_components >= 1.0:
            raise ValueError("reference.min_weight 与模态数的乘积必须小于 1")
        if self.error.ensemble_size < 2:
            raise ValueError("误差代理需要至少两个冻结集成成员")
        if self.error.features < 1 or self.error.relative_rmse < 0:
            raise ValueError("随机特征数必须为正，relative_rmse 不得为负")
        if self.error.frequency_scale <= 0:
            raise ValueError("随机特征频率尺度必须为正")
        if self.sampler.num_chains < 2 or self.sampler.num_steps < 1:
            raise ValueError("链数至少为 2，步数至少为 1")
        if self.sampler.initialization not in {"stationary", "single_mode", "overdispersed"}:
            raise ValueError("initialization 必须为 stationary、single_mode 或 overdispersed")
        if not 0.0 <= self.global_proposal.probability <= 1.0:
            raise ValueError("全局核概率必须位于 [0, 1]")
        if not 0.0 <= self.global_proposal.student_weight < 1.0:
            raise ValueError("学生 t 分量权重必须位于 [0, 1)")
        if self.adaptation.h_num <= 0 or self.adaptation.h_max <= 0:
            raise ValueError("步长上下界必须为正数")
        if self.adaptation.h_num >= self.adaptation.h_max:
            raise ValueError("h_num 必须严格小于 h_max")
        if self.adaptation.h0 <= 0 or self.adaptation.h_oper <= 0:
            raise ValueError("h0 和 h_oper 必须为正")
        if self.adaptation.h_oper < self.adaptation.h_num:
            raise ValueError("h_oper 不得小于 h_num")
        if self.adaptation.z_max <= 0:
            raise ValueError("z_max 必须为正")
        if self.adaptation.beta_energy < 0 or self.adaptation.beta_gradient < 0:
            raise ValueError("噪声门控斜率不得为负")
        if self.adaptation.tau_max < 1.0:
            raise ValueError("tau_max 不得小于 1")
        if self.adaptation.gamma < 0:
            raise ValueError("gamma 不得为负")
        if (
            min(
                self.adaptation.curvature_c,
                self.adaptation.curvature_safety,
                self.adaptation.epsilon_proposal,
                self.adaptation.drift_radius_factor,
                self.adaptation.epsilon,
            )
            <= 0
        ):
            raise ValueError("曲率、误差、漂移和数值稳定参数必须为正")
        if self.adaptation.curvature_mode not in {"exact", "power"}:
            raise ValueError("curvature_mode 必须为 exact 或 power")
        if self.adaptation.curvature_probes < 1 or self.adaptation.curvature_power_iterations < 1:
            raise ValueError("曲率探针数和幂迭代次数必须为正整数")
        if not 0.0 < self.calibration.fit_fraction < 1.0:
            raise ValueError("calibration.fit_fraction 必须位于 (0, 1)")
        if not 0.0 < self.calibration.calibration_fraction < 1.0:
            raise ValueError("calibration.calibration_fraction 必须位于 (0, 1)")
        if self.calibration.fit_fraction + self.calibration.calibration_fraction >= 1.0:
            raise ValueError("拟合集与校准集比例之和必须小于 1")
        if self.calibration.num_points < 20:
            raise ValueError("calibration.num_points 至少为 20")
        if not 0.0 < self.calibration.coverage < 1.0 or self.calibration.wide_scale <= 0:
            raise ValueError("校准覆盖率必须位于 (0, 1)，宽分布尺度必须为正")
        if self.sampler.diagnostic_stride < 1 or self.sampler.trajectory_stride < 1:
            raise ValueError("诊断与轨迹间隔必须为正整数")
        if self.sampler.crossing_stride < 1:
            raise ValueError("跨模态监测间隔必须为正整数")
        if self.sampler.recorded_chains < 1 or self.sampler.initialization_scale <= 0:
            raise ValueError("记录链数和初始化尺度必须为正")
        if self.metrics.fid_reference_repeats < 1 or self.metrics.min_rhat_draws < 4:
            raise ValueError("参考重复次数必须为正，且 min_rhat_draws 至少为 4")
        if self.metrics.mode_confirmation_checkpoints < 1:
            raise ValueError("模态确认次数必须为正整数")
        if not 0.0 < self.metrics.mode_confidence <= 1.0:
            raise ValueError("mode_confidence 必须位于 (0, 1]")
        if not 0.0 <= self.metrics.max_js_for_convergence <= math.log(2.0):
            raise ValueError("max_js_for_convergence 必须位于 [0, log(2)]")
        if self.metrics.min_mode_ratio_for_convergence < 0:
            raise ValueError("min_mode_ratio_for_convergence 不得为负")
        if self.output.reservoir_size < 1 or self.output.plot_grid_size < 2:
            raise ValueError("储备池容量必须为正，绘图网格边长至少为 2")
        if self.global_proposal.gaussian_scale <= 0 or self.global_proposal.student_scale <= 0:
            raise ValueError("全局提议尺度必须为正")
        if self.global_proposal.degrees_of_freedom <= 0:
            raise ValueError("学生 t 自由度必须为正")
        if (
            min(
                self.cost.energy_weight,
                self.cost.hessian_weight,
                self.cost.global_density_weight,
                self.cost.diffusion_weight,
            )
            < 0
        ):
            raise ValueError("成本权重不得为负")
        valid_modes = {"online", "offline", "disabled"}
        if self.wandb.mode not in valid_modes:
            raise ValueError(f"wandb.mode 必须属于 {sorted(valid_modes)}")
        from .methods import METHOD_REGISTRY

        unknown = set(self.sampler.methods) - set(METHOD_REGISTRY)
        if unknown:
            raise ValueError(f"未知方法编号: {sorted(unknown)}")
        checkpoints = sorted(
            {int(v) for v in self.sampler.checkpoints if 0 <= int(v) <= self.sampler.num_steps}
        )
        if 0 not in checkpoints:
            checkpoints.insert(0, 0)
        if self.sampler.num_steps not in checkpoints:
            checkpoints.append(self.sampler.num_steps)
        self.sampler.checkpoints = checkpoints

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTION_TYPES = {
    "target": TargetConfig,
    "whitening": WhiteningConfig,
    "reference": ReferenceConfig,
    "error": ErrorConfig,
    "calibration": CalibrationConfig,
    "adaptation": AdaptationConfig,
    "global_proposal": GlobalProposalConfig,
    "sampler": SamplerConfig,
    "metrics": MetricsConfig,
    "cost": CostConfig,
    "wandb": WandbConfig,
    "output": OutputConfig,
}


def _merge_dataclass(instance: Any, values: dict[str, Any], prefix: str) -> Any:
    allowed = {f.name for f in dataclasses.fields(instance)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"配置节 {prefix} 含未知字段: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"覆盖参数必须写成 路径=值: {raw}")
    key, value = raw.split("=", 1)
    return key.split("."), yaml.safe_load(value)


def _apply_override(data: dict[str, Any], raw: str) -> None:
    path, value = _parse_override(raw)
    cursor = data
    for part in path[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[path[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("配置文件顶层必须为映射")
    for override in overrides or []:
        _apply_override(data, override)

    top_allowed = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(data) - top_allowed
    if unknown:
        raise ValueError(f"顶层配置含未知字段: {sorted(unknown)}")

    cfg = ExperimentConfig()
    for key, value in data.items():
        if key in _SECTION_TYPES:
            if not isinstance(value, dict):
                raise ValueError(f"配置节 {key} 必须为映射")
            section = _SECTION_TYPES[key]()
            setattr(cfg, key, _merge_dataclass(section, value, key))
        else:
            setattr(cfg, key, value)
    cfg.validate()
    return cfg


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, allow_unicode=True, sort_keys=False)
