from __future__ import annotations

import torch

from ege_ah_mala.config import TargetConfig
from ege_ah_mala.distributions import build_target


def test_gmm_gradient_matches_autograd() -> None:
    target = build_target(
        TargetConfig(family="asymmetric_bimodal", dim=2, separation=2.0),
        dtype=torch.float64,
    )
    x = torch.tensor([[-1.2, 0.3], [0.1, -0.5], [2.4, 0.2]], dtype=torch.float64)
    x_auto = x.clone().requires_grad_(True)
    auto = torch.autograd.grad(target.energy(x_auto).sum(), x_auto)[0]
    analytic = target.grad_energy(x)
    torch.testing.assert_close(analytic, auto, rtol=1.0e-9, atol=1.0e-10)


def test_gmm_hessian_matches_autograd() -> None:
    target = build_target(
        TargetConfig(family="bimodal", dim=2, separation=2.0), dtype=torch.float64
    )
    points = torch.tensor([[-0.7, 0.4], [0.2, -0.3]], dtype=torch.float64)
    analytic = target.hessian_energy(points)
    automatic = []
    for point in points:
        value = point.clone().requires_grad_(True)
        hessian = torch.autograd.functional.hessian(
            lambda z: target.energy(z[None, :]).sum(), value
        )
        automatic.append(hessian)
    torch.testing.assert_close(
        analytic,
        torch.stack(automatic),
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_gmm_hessian_vector_product_matches_explicit_hessian() -> None:
    target = build_target(
        TargetConfig(family="highdim", dim=8, modes=3, highdim_mahalanobis=4.0),
        dtype=torch.float64,
    )
    points = target.sample(5, torch.Generator().manual_seed(37))
    vectors = torch.randn(5, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(38))
    expected = torch.einsum("nij,nj->ni", target.hessian_energy(points), vectors)
    actual = target.hessian_vector_product(points, vectors)
    torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-10)


def test_target_samples_are_reproducible() -> None:
    target = build_target(TargetConfig(family="ring", dim=2, modes=8), dtype=torch.float64)
    first = target.sample(16, torch.Generator().manual_seed(7))
    second = target.sample(16, torch.Generator().manual_seed(7))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
