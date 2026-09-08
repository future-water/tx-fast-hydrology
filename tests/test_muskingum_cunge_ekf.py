"""
Basically checks to make sure K, X are being updated properly
and that the Jacobian is being computed as expected from these updates
"""
import numpy as np
import pandas as pd

from tx_fast_hydrology.da import ExtendedKalmanFilter
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge
from tx_fast_hydrology.nutils import _aqat_par, _short_ts_aqat_par


def make_three_reach_model(assume_short_ts=None, sparse=False):
    timestamp = pd.Timestamp('2023-05-13T00:00:00Z')
    data = {
        'name': 'three-reach-confluence',
        'datetime': timestamp,
        'timedelta': pd.Timedelta(minutes=5),
        'reach_ids': ['headwater-a', 'headwater-b', 'outlet'],
        'startnodes': np.arange(3, dtype=np.int64),
        'endnodes': np.array([2, 2, 2], dtype=np.int64),
        'K': np.full(3, 300.0, dtype=np.float64),
        'X': np.full(3, 0.3, dtype=np.float64),
        'o_t': np.array([2.0, 1.0, 3.0], dtype=np.float64),
        'dx': np.full(3, 1000.0, dtype=np.float64),
    }
    geometry = {
        'So': np.full(3, 0.001),
        'dx': np.full(3, 1000.0),
        # The historical geometry key remains supported without replacing the
        # model's integer reach count.
        'n': np.full(3, 0.035),
        'Cs': np.full(3, 1.0),
        'Bw': np.full(3, 5.0),
        'Tw': np.full(3, 15.0),
        'TwCC': np.full(3, 30.0),
        'nCC': np.full(3, 0.06),
    }
    return MuskingumCunge(
        data, geometry=geometry, assume_short_ts=assume_short_ts,
        sparse=sparse,
    )


def dense_topological_transition(model):
    """Independent dense oracle for current-flow topological traversal."""
    transition = np.zeros((model.n, model.n), dtype=np.float64)
    remaining = model.indegree.copy()
    headwaters = model.startnodes[remaining == 0]
    for headwater in headwaters:
        reach = headwater
        while remaining[reach] == 0:
            transition[reach, reach] = model.chi[reach]
            downstream = model.endnodes[reach]
            if reach == downstream:
                break
            transition[downstream, reach] += model.beta[downstream]
            transition[downstream] += (
                model.alpha[downstream] * transition[reach]
            )
            remaining[downstream] -= 1
            reach = downstream
    return transition


def test_assume_short_ts_defaults_true_and_false_remains_available():
    assert make_three_reach_model().assume_short_ts is True
    assert make_three_reach_model(False).assume_short_ts is False


def test_mc_stores_dynamic_k_x_and_builds_synchronous_jacobian():
    model = make_three_reach_model()
    initial_time = model.datetime

    model.step(np.array([0.2, 0.1, 0.0]))

    assert model.n == 3
    assert model.mann_n.shape == (3,)
    assert model.datetime in model.K_history
    assert model.datetime in model.X_history
    assert len(model.K_history) == 2
    assert np.all(model.K >= model._coefficient_dt)
    assert np.all((model.X >= 0.0) & (model.X <= 0.5))
    assert not np.allclose(model.K_history[initial_time], model.K)

    np.testing.assert_allclose(
        model.alpha,
        model.compute_alpha(model.K, model.X, model._coefficient_dt),
    )
    np.testing.assert_allclose(
        model.beta,
        model.compute_beta(model.K, model.X, model._coefficient_dt),
    )
    np.testing.assert_allclose(
        model.chi,
        model.compute_chi(model.K, model.X, model._coefficient_dt),
    )

    F = model.state_transition_jacobian()
    expected = np.diag(model.chi)
    expected[2, 0] = model.alpha[2] + model.beta[2]
    expected[2, 1] = model.alpha[2] + model.beta[2]
    np.testing.assert_allclose(F, expected)

    model.load_state()
    assert model.datetime == initial_time
    np.testing.assert_allclose(model.K, np.full(3, 300.0))
    np.testing.assert_allclose(model.X, np.full(3, 0.3))
    np.testing.assert_allclose(model.depth, np.zeros(3))


def test_ekf_uses_latest_jacobian_for_covariance_and_reduces_residual():
    model = make_three_reach_model()
    timestamp = model.datetime
    measurements = pd.DataFrame(
        {'outlet': [3.0, 4.0]},
        index=[timestamp, timestamp + pd.Timedelta(minutes=10)],
    )
    P_init = np.eye(3)
    Q_cov = np.eye(3) * 0.01
    R_cov = np.eye(1) * 0.25
    ekf = ExtendedKalmanFilter(
        model, measurements, Q_cov=Q_cov, R_cov=R_cov,
        P_t_init=P_init,
    )
    model.bind_callback(ekf, key='ekf')

    # Production covariance propagation must not construct the dense F. The
    # explicit matrix is used only below as a test oracle.
    def fail_if_dense_jacobian_is_called():
        raise AssertionError('dense Jacobian called during EKF propagation')

    model.state_transition_jacobian = fail_if_dense_jacobian_is_called
    model.step(np.array([0.2, 0.1, 0.0]))

    F_test_only = np.diag(model.chi)
    F_test_only[2, 0] = model.alpha[2] + model.beta[2]
    F_test_only[2, 1] = model.alpha[2] + model.beta[2]
    expected_prior = F_test_only @ P_init @ F_test_only.T + Q_cov
    expected_prior = 0.5 * (expected_prior + expected_prior.T)
    np.testing.assert_allclose(ekf.P_t_prior, expected_prior)
    assert ekf.F is None

    observed = np.array([2])
    innovation_cov = expected_prior[np.ix_(observed, observed)] + R_cov
    expected_gain = np.linalg.solve(
        innovation_cov, expected_prior[observed, :]
    ).T
    expected_posterior = expected_prior - (
        expected_gain @ expected_prior[observed, :]
    )
    expected_posterior = 0.5 * (
        expected_posterior + expected_posterior.T
    )
    np.testing.assert_allclose(ekf.P_t_next, expected_posterior)
    np.testing.assert_allclose(ekf.P_t_next, ekf.P_t_next.T)
    assert np.linalg.eigvalsh(ekf.P_t_next).min() >= -1e-12

    interpolated_observation = 3.5
    posterior_residual = interpolated_observation - model.o_t_next[2]
    assert abs(posterior_residual) < abs(ekf.dz[0])
    assert ekf.datetime in ekf.covariance_trace_history

def test_matrix_free_mc_covariance_matches_dense_test_oracle():
    """The dense comparison belongs in tests, never in filter execution."""
    model = make_three_reach_model()
    model.step(np.array([0.2, 0.1, 0.0]))

    generator = np.random.default_rng(24601)
    factor = generator.normal(size=(model.n, model.n))
    covariance = factor @ factor.T
    out = np.empty_like(covariance)

    actual = _short_ts_aqat_par(
        covariance, out, model.endnodes,
        model.alpha, model.beta, model.chi,
    )
    F_test_only = model.state_transition_jacobian()
    expected = F_test_only @ covariance @ F_test_only.T

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1e-13)


def test_topological_routing_and_matrix_free_covariance_match_dense_oracle():
    model = make_three_reach_model(assume_short_ts=False)
    model.step(np.array([0.2, 0.1, 0.0]))

    np.testing.assert_allclose(
        model.i_t_next[2], model.o_t_next[:2].sum(),
    )
    expected_transition = dense_topological_transition(model)
    np.testing.assert_allclose(
        model.state_transition_jacobian(), expected_transition,
    )

    generator = np.random.default_rng(1915)
    factor = generator.normal(size=(model.n, model.n))
    covariance = factor @ factor.T
    headwaters = model.startnodes[model.indegree == 0]
    actual = _aqat_par(
        covariance, np.empty_like(covariance), headwaters, model.endnodes,
        model.alpha, model.beta, model.chi, model.indegree,
    )
    expected = expected_transition @ covariance @ expected_transition.T
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_ekf_ignores_missing_measurements_without_changing_dimensions():
    model = make_three_reach_model()
    timestamp = model.datetime
    measurements = pd.DataFrame(
        {
            'headwater-a': [np.nan, np.nan],
            'outlet': [3.0, 4.0],
        },
        index=[timestamp, timestamp + pd.Timedelta(minutes=5)],
    )
    ekf = ExtendedKalmanFilter(
        model,
        measurements,
        Q_cov=np.eye(model.n) * 0.01,
        R_cov=np.eye(2) * 0.25,
        P_t_init=np.eye(model.n),
    )
    model.bind_callback(ekf, key='ekf')

    model.step(np.array([0.2, 0.1, 0.0]))

    np.testing.assert_array_equal(ekf.observation_indices, [2])
    assert ekf.K.shape == (model.n, 1)
    assert np.isfinite(ekf.P_t_next).all()
    assert np.isfinite(model.o_t_next).all()
