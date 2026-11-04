"""Small dense Kalman oracles and missing-observation regression contracts."""

import numpy as np
import pandas as pd
import pytest

from tx_fast_hydrology.da import KalmanFilter, ExtendedKalmanFilter


# Exercise both MC routing modes and one linear KF configuration.
FILTER_CASES = [
    pytest.param('linear', KalmanFilter, None, id='linear-kf'),
    pytest.param('mc', ExtendedKalmanFilter, False, id='mc-topological'),
    pytest.param('mc', ExtendedKalmanFilter, True, id='mc-short-timestep'),
]


def observations(model, values=None):
    return pd.DataFrame(
        {'3': [3.0, 4.0, 5.0] if values is None else values},
        index=pd.date_range(model.datetime, periods=3, freq='5min'),
    )


@pytest.mark.parametrize('model_name,filter_type,short_ts', FILTER_CASES)
def test_filter_matches_dense_oracle(
    request, model_name, filter_type, short_ts
):
    model = request.getfixturevalue(model_name)
    if model_name == 'mc':
        model.assume_short_ts = short_ts
    initial_cov = np.array([[2.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 3.0]])
    process = np.eye(3) * 0.02
    noise = np.array([[0.4]])
    filt = filter_type(model, observations(model), process, noise, initial_cov)
    model.step(np.array([0.2, 0.1, 0.0]))
    before = model.o_t_next.copy()
    # Use a dense reference here; production covariance stays matrix-free.
    transition = (
        model.state_transition_jacobian() if model_name == 'mc' else model.A
    )
    prior = transition @ initial_cov @ transition.T + process
    gain = np.linalg.solve(prior[2:, 2:] + noise, prior[2:, :]).T
    innovation = np.array([4.0 - before[2]])
    filt.filter()
    np.testing.assert_allclose(filt.K, gain, atol=1e-12)
    np.testing.assert_allclose(model.o_t_next, before + gain @ innovation)
    np.testing.assert_allclose(
        filt.P_t_next, prior - gain @ prior[2:, :], atol=1e-12
    )
    np.testing.assert_allclose(filt.P_t_next, filt.P_t_next.T, atol=1e-12)
    assert np.linalg.eigvalsh(filt.P_t_next).min() >= -1e-12


@pytest.mark.parametrize('model_name,filter_type,short_ts', FILTER_CASES)
def test_filter_all_missing_is_prediction_only(
    request, model_name, filter_type, short_ts
):
    model = request.getfixturevalue(model_name)
    if model_name == 'mc':
        model.assume_short_ts = short_ts
    filt = filter_type(
        model,
        observations(model, [np.nan] * 3),
        np.eye(3) * 0.01,
        np.eye(1),
        np.eye(3),
    )
    model.step(np.zeros(3))
    before = model.o_t_next.copy()
    inflow = model.i_t_next.copy()
    transition = (
        model.state_transition_jacobian() if model_name == 'mc' else model.A
    )
    # Prediction must still advance covariance when no gauge is available.
    expected = transition @ np.eye(3) @ transition.T + np.eye(3) * 0.01
    filt.filter()
    np.testing.assert_array_equal(model.o_t_next, before)
    np.testing.assert_array_equal(model.i_t_next, inflow)
    np.testing.assert_allclose(filt.P_t_next, expected)
    assert filt.K.shape == (3, 0)
    assert filt.observation_indices.size == 0


@pytest.mark.parametrize(
    'model_name,filter_type',
    [('linear', KalmanFilter), ('mc', ExtendedKalmanFilter)],
)
def test_filter_partial_missing_with_reordered_correlated_noise(
    request, model_name, filter_type, caplog
):
    model = request.getfixturevalue(model_name)
    # Reverse the gauge order to check the noise-covariance permutation too.
    frame = pd.DataFrame(
        {'3': [4.0, 5.0], '1': [np.nan, np.nan]},
        index=[model.datetime, model.datetime + model.timedelta],
    )
    noise = np.array([[0.4, 0.1], [0.1, 0.8]])
    filt = filter_type(model, frame, np.eye(3) * 0.01, noise, np.eye(3))
    np.testing.assert_allclose(filt.R_cov, noise[::-1, ::-1])
    assert 'Permuting columns' in caplog.text
    transition = (
        model.state_transition_jacobian() if model_name == 'mc' else model.A
    )
    prior = transition @ transition.T + np.eye(3) * 0.01
    expected_gain = np.linalg.solve(prior[2:, 2:] + 0.4, prior[2:, :]).T
    filt.filter()
    np.testing.assert_array_equal(filt.observation_indices, [2])
    np.testing.assert_allclose(filt.K, expected_gain)
    np.testing.assert_allclose(
        filt.P_t_next, prior - expected_gain @ prior[2:, :]
    )
    assert filt.K.shape == (3, 1)
    assert np.isfinite(model.o_t_next).all()


@pytest.mark.parametrize('argument', ['Q_cov', 'R_cov', 'P_t_init'])
def test_ekf_covariance_shape_validation(mc, argument):
    kwargs = dict(Q_cov=np.eye(3), R_cov=np.eye(1), P_t_init=np.eye(3))
    kwargs[argument] = np.zeros((2, 2))
    with pytest.raises(ValueError, match=argument):
        ExtendedKalmanFilter(mc, observations(mc), **kwargs)


@pytest.mark.parametrize(
    'model_name,filter_type',
    [('linear', KalmanFilter), ('mc', ExtendedKalmanFilter)],
)
def test_filter_interpolation_and_end_of_observations(
    request, model_name, filter_type
):
    model = request.getfixturevalue(model_name)
    frame = observations(model)
    filt = filter_type(model, frame, np.eye(3) * 0.01, np.eye(1), np.eye(3))
    middle = model.datetime + pd.Timedelta(minutes=2)
    np.testing.assert_allclose(filt.interpolate_input(middle), [3.4])
    np.testing.assert_allclose(
        filt.interpolate_input(middle, method='nearest'), [3.0]
    )
    with pytest.raises(ValueError):
        filt.interpolate_input(middle, method='cubic')
    np.testing.assert_array_equal(filt.latest_measurement, [5.0])
    filt.__on_simulation_start__()
    filt.save_state()
    saved_covariance = filt.P_t_next.copy()
    # After the observation window, the callback should leave covariance alone.
    model.datetime = frame.index[-1] + model.timedelta
    filt.__on_step_end__()
    filt.__on_simulation_start__()
    np.testing.assert_array_equal(filt.P_t_next, saved_covariance)
    filt.P_t_next = np.eye(3) * 10
    filt.load_state()
    np.testing.assert_array_equal(filt.P_t_next, saved_covariance)
