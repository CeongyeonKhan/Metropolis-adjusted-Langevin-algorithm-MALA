from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    corrected: bool
    global_kernel: bool
    energy_adaptation: bool
    curvature_constraint: bool
    error_constraint: str
    adaptive_noise: bool
    drift_constraint: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "B0": MethodSpec("B0", False, False, False, False, "none", False, False),
    "B1": MethodSpec("B1", True, False, False, False, "none", False, False),
    "B2": MethodSpec("B2", True, True, False, False, "none", False, False),
    "A1": MethodSpec("A1", True, False, True, True, "none", False, True),
    "A2-O": MethodSpec("A2-O", True, False, True, True, "oracle", False, True),
    "A2-P": MethodSpec("A2-P", True, False, True, True, "proxy", False, True),
    "A3": MethodSpec("A3", True, False, True, True, "proxy", True, True),
    "A4": MethodSpec("A4", True, True, True, True, "proxy", True, True),
    "A4-NC": MethodSpec("A4-NC", False, True, True, True, "proxy", True, True),
}


def get_method_spec(name: str) -> MethodSpec:
    try:
        return METHOD_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"未知方法编号 {name}") from exc
