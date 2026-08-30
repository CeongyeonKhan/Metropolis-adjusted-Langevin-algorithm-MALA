from __future__ import annotations

import torch
from conftest import make_test_system

from ege_ah_mala.adaptation import isotropic_log_proposal
from ege_ah_mala.config import GlobalProposalConfig
from ege_ah_mala.global_proposal import GlobalMixtureProposal
from ege_ah_mala.methods import get_method_spec


def test_general_gamma_local_detailed_balance_identity() -> None:
    target, _, controller = make_test_system(gamma=0.0)
    method = get_method_spec("A3")
    x = torch.tensor([[-1.5, 0.2], [1.2, -0.3]], dtype=torch.float64)
    y = torch.tensor([[-0.9, 0.4], [0.8, 0.1]], dtype=torch.float64)
    state_x = controller.evaluate(x, target.energy(x), method)
    state_y = controller.evaluate(y, target.energy(y), method)
    log_q_y_x = isotropic_log_proposal(y, state_x)
    log_q_x_y = isotropic_log_proposal(x, state_y)
    log_ratio_xy = -target.energy(y) + target.energy(x) + log_q_x_y - log_q_y_x
    log_ratio_yx = -log_ratio_xy
    log_flux_xy = (
        -target.energy(x) + log_q_y_x + torch.minimum(torch.zeros_like(log_ratio_xy), log_ratio_xy)
    )
    log_flux_yx = (
        -target.energy(y) + log_q_x_y + torch.minimum(torch.zeros_like(log_ratio_yx), log_ratio_yx)
    )
    torch.testing.assert_close(log_flux_xy, log_flux_yx, rtol=1.0e-10, atol=1.0e-10)


def test_gamma_zero_does_not_scale_drift_by_tau() -> None:
    target, _, controller = make_test_system(gamma=0.0)
    method = get_method_spec("A3")
    x = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    state = controller.evaluate(x, target.energy(x), method)
    expected_mean = x - state.h[:, None] * state.gradient
    variance = 2.0 * state.h * state.tau
    destination = expected_mean.clone()
    expected_log = -0.5 * target.dim * torch.log(2.0 * torch.pi * variance)
    actual_log = isotropic_log_proposal(destination, state)
    torch.testing.assert_close(actual_log, expected_log, rtol=1.0e-10, atol=1.0e-10)


def test_step_does_not_exceed_active_bounds_except_numeric_floor() -> None:
    target, _, controller = make_test_system(gamma=1.0)
    method = get_method_spec("A4")
    x = target.sample(32, torch.Generator().manual_seed(18))
    state = controller.evaluate(x, target.energy(x), method)
    bounds = torch.stack([state.h_energy, state.h_curvature, state.h_error, state.h_drift], dim=1)
    safe = ~state.numeric_floor
    assert bool((state.h[safe, None] <= bounds[safe] + 1.0e-12).all())
    assert bool((state.h <= controller.config.h_max + 1.0e-12).all())


def test_ensemble_jacobian_vector_product_matches_explicit_jacobian() -> None:
    target, ensemble, _ = make_test_system(gamma=1.0)
    points = target.sample(6, torch.Generator().manual_seed(23))
    vectors = torch.randn(
        6,
        target.dim,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(24),
    )
    expected = torch.einsum("nij,nj->ni", ensemble.jacobian_mean(points), vectors)
    actual = ensemble.jacobian_vector_product_mean(points, vectors)
    torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-10)


def test_global_gaussian_component_matches_torch_distribution() -> None:
    target, _, _ = make_test_system(relative_rmse=0.0)
    config = GlobalProposalConfig(
        gaussian_scale=2.0,
        student_weight=0.0,
    )
    proposal = GlobalMixtureProposal(target, config)
    x = torch.tensor([[-1.0, 0.2], [0.5, -0.7]], dtype=torch.float64)
    component_values = []
    for weight, mean, covariance in zip(target.weights, target.means, target.covariances):
        distribution = torch.distributions.MultivariateNormal(
            mean, covariance_matrix=config.gaussian_scale * covariance
        )
        component_values.append(torch.log(weight) + distribution.log_prob(x))
    expected = torch.logsumexp(torch.stack(component_values, dim=1), dim=1)
    torch.testing.assert_close(proposal.log_prob(x), expected, rtol=1.0e-10, atol=1.0e-10)


def test_global_proposal_is_reproducible_and_finite() -> None:
    target, _, _ = make_test_system(relative_rmse=0.0)
    proposal = GlobalMixtureProposal(target, GlobalProposalConfig())
    first = proposal.sample(32, torch.Generator().manual_seed(19))
    second = proposal.sample(32, torch.Generator().manual_seed(19))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(proposal.log_prob(first)).all())
