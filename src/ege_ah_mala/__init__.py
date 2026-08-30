"""能量—梯度误差约束自适应混合 MALA。"""

from .config import ExperimentConfig, load_config
from .distributions import GaussianMixtureTarget, build_target
from .methods import MethodSpec, get_method_spec

__all__ = [
    "ExperimentConfig",
    "GaussianMixtureTarget",
    "MethodSpec",
    "build_target",
    "get_method_spec",
    "load_config",
]

__version__ = "0.1.0"
