"""Nudging input validation, small parameter files, QC, and restarts."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tx_fast_hydrology.nudging import (
    StreamflowNudging,
    WRFHydroStreamflowNudging,
)


NUDGERS = [StreamflowNudging, WRFHydroStreamflowNudging]


def measurements(model):
    return pd.DataFrame(
        {'1': [4.0, 8.0]},
        index=[model.datetime, model.datetime + model.timedelta],
    )


@pytest.mark.parametrize('kind', NUDGERS)
@pytest.mark.parametrize(
    'changes,error',
    [
        ({'reach_ids': []}, ValueError),
        ({'reach_ids': ['absent']}, ValueError),
        ({'quality': -1.0}, ValueError),
        ({'quality': pd.DataFrame({'1': [1.0]})}, TypeError),
    ],
)
def test_nudging_rejects_bad_mapping_and_quality(linear, kind, changes, error):
    with pytest.raises(error):
        kind(linear, measurements(linear), **changes)


@pytest.mark.parametrize('kind', NUDGERS)
def test_nudging_rejects_duplicate_reach_mapping(linear, kind):
    frame = measurements(linear).rename(columns={'1': 'a'})
    frame['b'] = 5.0
    with pytest.raises(ValueError, match='one stream gage'):
        kind(linear, frame, reach_ids=['1', '1'])


@pytest.mark.parametrize('kind', NUDGERS)
def test_nudging_requires_timestamped_dataframe(linear, kind):
    frame = pd.DataFrame({'1': [1.0]})
    with pytest.raises(TypeError):
        kind(linear, frame)


@pytest.mark.parametrize('kind', NUDGERS)
def test_normalization_deduplicates_times_and_rejects_duplicate_columns(
    linear, kind
):
    timestamp = linear.datetime.tz_localize(None)
    frame = pd.DataFrame(
        {'1': [2.0, 6.0, 4.0]},
        index=[timestamp, timestamp, timestamp + linear.timedelta],
    )
    callback = kind(linear, frame)
    assert callback.measurements.iloc[0, 0] == 6.0
    assert str(callback.measurements.index.tz) == 'UTC'
    duplicate = pd.concat([frame, frame], axis=1)
    with pytest.raises(ValueError, match='columns must be unique'):
        kind(linear, duplicate)


@pytest.mark.parametrize('kind', NUDGERS)
def test_dataframe_quality_alignment_and_idempotence(linear, kind):
    frame = measurements(linear)
    quality = pd.DataFrame(
        {'1': [1.0, 0.0]}, index=frame.index.tz_localize(None)
    )
    callback = kind(linear, frame, quality=quality)
    linear.bind_callback(callback)
    callback.__on_simulation_start__()
    initial = linear.o_t_next.copy()
    # A second update at the same model time must not apply the nudge twice.
    callback.update()
    np.testing.assert_array_equal(linear.o_t_next, initial)
    linear.save_state()
    linear.step(np.zeros(3))
    linear.load_state()
    np.testing.assert_array_equal(linear.o_t_next, initial)
    assert callback._last_applied_time == frame.index[0]
    callback.clear_assimilation_end()
    callback.set_assimilation_end('2023-05-13')
    assert callback.assimilation_end == frame.index[0]
    callback.set_assimilation_end(frame.index[0].tz_convert('US/Central'))
    assert callback.assimilation_end == frame.index[0]


@pytest.mark.parametrize(
    'kind,argument',
    [
        (StreamflowNudging, 'decay_coefficient'),
        (WRFHydroStreamflowNudging, 'tau_minutes'),
        (WRFHydroStreamflowNudging, 'G'),
    ],
)
@pytest.mark.parametrize('bad', [[1.0, 2.0], [np.nan]])
def test_nudging_per_gage_parameter_shapes(linear, kind, argument, bad):
    with pytest.raises(ValueError):
        kind(linear, measurements(linear), **{argument: bad})


@pytest.mark.parametrize(
    'kind,argument',
    [
        (StreamflowNudging, 'decay_coefficient'),
        (WRFHydroStreamflowNudging, 'tau_minutes'),
    ],
)
def test_nudging_positive_time_constants(linear, kind, argument):
    with pytest.raises(ValueError, match='positive'):
        kind(linear, measurements(linear), **{argument: 0.0})


@pytest.mark.parametrize(
    'changes,error',
    [
        ({'quality_threshold': 2.0}, ValueError),
        ({'interpolation_limit_minutes': 0}, ValueError),
        ({'last_observation_values': [2.0]}, ValueError),
        (
            {
                'last_observation_values': [2.0, 3.0],
                'last_observation_times': ['2023-05-13'],
            },
            ValueError,
        ),
        (
            {'last_observation_values': [2.0], 'last_observation_times': []},
            ValueError,
        ),
        (
            {
                'last_observation_values': [2.0],
                'last_observation_times': [pd.NaT],
            },
            ValueError,
        ),
    ],
)
def test_troute_restart_and_interpolation_validation(linear, changes, error):
    with pytest.raises(error):
        StreamflowNudging(linear, measurements(linear), **changes)


def test_troute_restart_persistence_without_observations(linear):
    empty = pd.DataFrame({'1': []}, index=pd.DatetimeIndex([], tz='UTC'))
    # Seed the last observation so persistence works with an empty table.
    callback = StreamflowNudging(
        linear,
        empty,
        last_observation_values=[6.0],
        last_observation_times=['2023-05-12T23:00:00'],
    )
    callback.update()
    assert linear.o_t_next[0] == pytest.approx(2.0 + 4.0 * np.exp(-0.5))
    assert not callback.measurements.size


@pytest.mark.parametrize(
    'thresholds', [[10.0], np.full((12, 1), 10.0), np.full((1, 12, 1), 10.0)]
)
def test_wrf_threshold_broadcasting_and_persistence_category(
    linear, thresholds
):
    # The modeled flow selects the decay category below the threshold.
    coefficients = np.tile([30.0, 120.0], (12, 1))
    callback = WRFHydroStreamflowNudging(
        linear,
        measurements(linear),
        q_thresholds=thresholds,
        exp_coefficients=coefficients,
        assimilation_end=linear.datetime,
    )
    callback.update()
    callback.set_assimilation_end(linear.datetime.tz_localize(None))
    linear.datetime += pd.Timedelta(minutes=60)
    linear.o_t_next[0] = 2.0
    callback.update()
    assert linear.o_t_next[0] == pytest.approx(2.0 + 2.0 * np.exp(-2.0))
    assert (
        callback._persistence_nudge(
            0, linear.datetime - pd.Timedelta(days=1), 2.0
        )
        == 0.0
    )


@pytest.mark.parametrize(
    'changes',
    [
        {'q_thresholds': np.ones((2, 2))},
        {'q_thresholds': [np.nan]},
        {'exp_coefficients': np.ones((2, 2))},
        {'exp_coefficients': 0.0},
        {'minimum_flow': -1.0},
    ],
)
def test_wrf_parameter_validation(linear, changes):
    with pytest.raises(ValueError):
        WRFHydroStreamflowNudging(linear, measurements(linear), **changes)


@pytest.mark.parametrize('persist', [False, True])
def test_wrf_missing_without_history_and_minimum_flow(linear, persist):
    frame = measurements(linear) * np.nan
    callback = WRFHydroStreamflowNudging(
        linear, frame, temporal_persistence=persist
    )
    callback.update()
    np.testing.assert_allclose(linear.o_t_next, [2.0, 1.0, 3.0])
    other = WRFHydroStreamflowNudging(
        linear, measurements(linear) * 0.0, minimum_flow=0.5
    )
    other.update()
    assert linear.o_t_next[0] == pytest.approx(0.5)


def test_nwm_nudging_parameters_file(tmp_path, linear):
    path = tmp_path / 'nudgingParams.nc'
    # Put another station first to verify lookup by station ID.
    xr.Dataset(
        {
            'stationId': ('gage', np.array([b'other', b'station '])),
            'G': ('gage', [0.5, 0.8]),
            'tau': ('gage', [10.0, 15.0]),
            'qThresh': (
                ('gage', 'month', 'threshold'),
                np.full((2, 12, 1), -100.0),
            ),
            'expCoeff': (
                ('gage', 'month', 'category'),
                np.full((2, 12, 2), 120.0),
            ),
        }
    ).to_netcdf(path, engine='h5netcdf')
    frame = measurements(linear).rename(columns={'1': 'station'})
    callback = WRFHydroStreamflowNudging.from_nwm_parameters(
        linear, frame, path, reach_ids=['1']
    )
    np.testing.assert_array_equal(callback.G, [0.8])
    callback.update()
    assert linear.o_t_next[0] == pytest.approx(3.6)
    with pytest.raises(ValueError, match='one entry'):
        WRFHydroStreamflowNudging.from_nwm_parameters(
            linear, frame, path, station_ids=[]
        )
    with pytest.raises(ValueError, match='absent'):
        WRFHydroStreamflowNudging.from_nwm_parameters(
            linear, frame, path, station_ids=['absent']
        )
