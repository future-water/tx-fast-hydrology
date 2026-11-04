"""Channel/reservoir collection boundaries and legacy file readers."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tx_fast_hydrology.da import (
    KalmanFilter,
    ExtendedKalmanFilter,
    ReservoirNudging,
)
from tx_fast_hydrology.io import ModelEncoder
from tx_fast_hydrology.muskingum import (
    Connection,
    ModelCollection,
    Muskingum,
    Reservoir,
    load_reservoir_model_file,
)
from tx_fast_hydrology.simulation import Simulation, AsyncSimulation


# Dummy file data
@pytest.fixture
def reservoir_data():
    return {
        'name': 'lake',
        'datetime': pd.Timestamp('2023-05-13', tz='UTC'),
        'timedelta': pd.Timedelta(minutes=5),
        'reach_ids': ['inlet', 'outlet'],
        'reservoir_id': ['lake'],
        'outlet_index': 1,
        **{
            key: np.array([value])
            for key, value in {
                'A_s': 1e6,
                'C_w': 1.6,
                'L': 20.0,
                'L_d': 5.0,
                'h_max': 5.0,
                'h_w': 3.0,
                'h_o': 1.0,
                'C_o': 0.6,
                'O_a': 2.0,
                'h_t': 2.0,
            }.items()
        },
        'o_t': np.array([0.0, 4.0]),
    }


def test_reservoir_json_loader_and_callback_lifecycle(
    tmp_path, reservoir_data
):
    # Dump the dummy data
    path = tmp_path / 'lake.json'
    path.write_text(
        json.dumps(reservoir_data, cls=ModelEncoder), encoding='utf-8'
    )
    # Load it back in
    loaded = load_reservoir_model_file(path)
    lake = Reservoir(loaded, load_optional=True)
    lake.load_model_file(path)
    forecast = pd.DataFrame(
        {'outlet': [4.0, 8.0]},
        index=[lake.datetime, lake.datetime + lake.timedelta],
    )
    # Run da
    callback = ReservoirNudging(lake, forecast)
    lake.bind_callback(callback)
    callback.__on_simulation_start__()
    # Test next measurement
    np.testing.assert_array_equal(callback.latest_measurement, [8.0])
    np.testing.assert_array_equal(
        callback.interpolate_input(lake.datetime, method='nearest'), [4.0]
    )
    with pytest.raises(ValueError):
        callback.interpolate_input(lake.datetime, method='cubic')

    lake.save_state()
    # Step once and make sure da has replaced data
    inputs = pd.DataFrame(1.0, index=forecast.index, columns=lake.reach_ids)
    list(
        lake.simulate_iter(
            inputs,
            start_time=lake.datetime,
            end_time=forecast.index[-1],
            o_t_init=np.array([0.0, 4.0]),
        )
    )
    assert lake.o_t_next[1] == 8.0
    # Show loading restores time
    lake.load_state()
    assert lake.datetime == forecast.index[0]
    lake.datetime = forecast.index[-1] + lake.timedelta
    callback.__on_simulation_start__()
    assert lake.unbind_callback('callback') is callback
    lake.init_states(i_t_next=np.array([1.0, 2.0]), h_t_next=np.array([1.0]))
    lake.step(np.ones(2), timedelta=pd.Timedelta(seconds=60))
    assert np.isfinite(lake.h_t_next).all()
    for keyword in ('i_t_next', 'h_t_next'):
        with pytest.raises(ValueError):
            lake.init_states(**{keyword: np.zeros(3)})


@pytest.mark.parametrize(
    'mutation,error',
    [
        (lambda data: data.pop('A_s'), ValueError),
        (lambda data: data.update(C_w=np.ones(2)), ValueError),
    ],
)
def test_reservoir_model_validation(reservoir_data, mutation, error):
    mutation(reservoir_data)
    with pytest.raises(error):
        Reservoir(reservoir_data)


def test_reservoir_lateral_nan_interpolation(reservoir_data):
    lake = Reservoir(reservoir_data)
    oracle = Reservoir(dict(reservoir_data))
    inputs = pd.DataFrame(
        {'inlet': [1.0, np.nan, 3.0], 'outlet': [np.nan, 2.0, np.nan]},
        index=pd.date_range(lake.datetime, periods=3, freq='5min').as_unit(
            'us'
        ),
    )
    original = inputs.copy(deep=True)
    # These are the expected inflows at the two five-minute routing steps.
    expected = []
    for forcing in ([2.0, 2.0], [3.0, 2.0]):
        oracle.step(np.array(forcing))
        expected.append(oracle.h_t_next.copy())
    actual = [state.h_t_next.copy() for state in lake.simulate_iter(inputs)]
    np.testing.assert_allclose(actual, expected, rtol=1e-3)
    pd.testing.assert_frame_equal(inputs, original)
    inputs['inlet'] = np.nan
    with pytest.raises(ValueError, match='finite.*inlet'):
        list(
            lake.simulate_iter(inputs, end_time=lake.datetime + lake.timedelta)
        )


def test_reservoir_bad_file_fields_and_lengths(tmp_path, reservoir_data):
    path = tmp_path / 'bad.json'
    for payload in ({}, dict(reservoir_data, C_w=np.ones(2))):
        path.write_text(
            json.dumps(payload, cls=ModelEncoder), encoding='utf-8'
        )
        with pytest.raises(ValueError):
            load_reservoir_model_file(path)


@pytest.mark.parametrize('load_optional', [False, True])
@pytest.mark.parametrize('path_type', [str, Path])
def test_reservoir_constructor_json_path(
    tmp_path, reservoir_data, load_optional, path_type
):
    path = tmp_path / 'lake.json'
    path.write_text(
        json.dumps(reservoir_data, cls=ModelEncoder), encoding='utf-8'
    )
    lake = Reservoir(path_type(path), load_optional=load_optional)
    np.testing.assert_array_equal(lake.o_t_next, reservoir_data['o_t'])
    np.testing.assert_array_equal(lake.h_t_next, reservoir_data['h_t'])
    lake.step(np.ones(lake.n))
    assert np.isfinite(lake.o_t_next).all()


@pytest.mark.parametrize('field', ['datetime', 'timedelta'])
def test_collection_requires_synchronized_models(linear, field):
    other = linear.copy()
    other.name = 'other'
    collection = ModelCollection([linear, other], name='pair')
    assert getattr(collection, field) == getattr(linear, field)
    setattr(other, field, getattr(other, field) + pd.Timedelta(seconds=60))
    with pytest.raises(ValueError, match='same'):
        getattr(collection, field)


def test_collection_checkpoint_and_simulation_proxies(linear):
    collection = ModelCollection([linear])
    frame = pd.DataFrame(
        0.0,
        index=pd.date_range(linear.datetime, periods=2, freq='5min'),
        columns=linear.reach_ids,
    )
    simulation = Simulation(collection, frame)
    simulation.init_states(pd.Series([1.0, 2.0, 3.0], index=linear.reach_ids))
    simulation.save_states()
    initial = linear.o_t_next.copy()
    linear.step(np.zeros(3))
    simulation.load_states()
    np.testing.assert_array_equal(linear.o_t_next, initial)
    simulation.set_datetime(frame.index[-1])
    assert simulation.datetime == frame.index[-1]
    assert collection.timedelta == pd.Timedelta(minutes=5)


def test_legacy_geojson_reader(tmp_path):
    path = tmp_path / 'network.json'
    payload = {
        'features': [
            {
                'attributes': {
                    'COMID': rid,
                    'toCOMID': to,
                    'Shape_Length': 1000.0,
                },
                'geometry': {'paths': [[[0.0, 0.0], [1.0, 1.0]]]},
            }
            for rid, to in [(1, 3), (2, 3), (3, 0)]
        ]
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    model = Muskingum.from_nhd_geojson(path)
    assert model.reach_ids == ['1', '2', '3']
    np.testing.assert_array_equal(model.endnodes, [2, 2, 2])
    # Node indices must stay int64 on Windows as well as Linux.
    assert model.startnodes.dtype == np.int64
    assert model.endnodes.dtype == np.int64


def test_collection_json_without_explicit_model_type(tmp_path, linear):
    path = tmp_path / 'old.json'
    payload = {
        'models': {linear.name: {'model': linear.info}},
        'connections': {},
    }
    path.write_text(json.dumps(payload, cls=ModelEncoder), encoding='utf-8')
    assert isinstance(
        ModelCollection.from_file(path).models[linear.name], Muskingum
    )


def test_async_mc_boundary_transfer(linear, mc):
    linear.name, mc.name = 'up', 'down'
    # Distinct identifiers allow one shared input frame.
    mc.reach_ids = ['4', '5', '6']
    connection = Connection(linear, mc, 2, 0, name='boundary')
    linear.sinks.append(connection)
    mc.sources.append(connection)
    collection = ModelCollection([linear, mc])
    frame = pd.DataFrame(
        0.0,
        index=pd.date_range(linear.datetime, periods=3, freq='5min'),
        columns=linear.reach_ids + mc.reach_ids,
    )
    simulation = AsyncSimulation(collection, frame)
    output = asyncio.run(simulation.simulate())
    np.testing.assert_allclose(
        simulation.inputs['down']['4'].iloc[1:], output['up']['3'].iloc[1:]
    )
    assert output['down'].shape == (3, 3)


@pytest.mark.parametrize(
    'with_troute', [False, True], ids=['routing', 'troute']
)
@pytest.mark.parametrize(
    'model_name,filter_type,short_ts',
    [
        pytest.param('linear', KalmanFilter, None, id='linear-kf'),
        pytest.param('mc', ExtendedKalmanFilter, False, id='mc-topological'),
        pytest.param('mc', ExtendedKalmanFilter, True, id='mc-short-timestep'),
    ],
)
def test_async_single_step_routing_and_filtering(
    request, model_name, filter_type, short_ts, with_troute
):
    model = request.getfixturevalue(model_name)
    if model_name == 'mc':  # because linear muskingum has no short_ts option
        model.assume_short_ts = short_ts
    oracle = model.copy()
    start = model.datetime
    inputs = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.2, 0.1, 0.3], [0.4, 0.2, 0.1]],
        index=pd.date_range(start, periods=3, freq='5min'),
        columns=model.reach_ids,
    )
    if with_troute:
        # External routing supplies discharge; assimilation needs no laterals.
        inputs.loc[:, :] = np.nan
    measurements = pd.DataFrame({'3': [5.0, 6.0, 7.0]}, index=inputs.index)
    filt, oracle_filter = (
        filter_type(
            channel,
            measurements,
            np.eye(3) * 0.01,
            np.array([[0.4]]),
            np.eye(3),
        )
        for channel in (model, oracle)
    )
    model.bind_callback(filt)
    simulation = AsyncSimulation(
        ModelCollection([model]), inputs, with_troute=with_troute
    )

    # Repeated calls must use the current state and assimilate exactly once.
    for step in range(1, 3):
        previous = oracle.o_t_next.copy()
        previous_time = oracle.datetime
        if not with_troute:
            oracle.step(inputs.iloc[step].to_numpy())
        oracle_filter.filter()
        filtered_time = oracle.datetime
        expected_times = (
            [filtered_time] if with_troute else [previous_time, filtered_time]
        )
        expected_flows = (
            [oracle.o_t_next] if with_troute else [previous, oracle.o_t_next]
        )

        output = asyncio.run(simulation.simulate(only_one_time_step=True))
        frame = output[model.name]
        assert frame.columns.tolist() == model.reach_ids
        pd.testing.assert_index_equal(
            frame.index, pd.DatetimeIndex(expected_times)
        )
        np.testing.assert_allclose(frame.to_numpy(), expected_flows)
        np.testing.assert_allclose(model.i_t_next, oracle.i_t_next)
        np.testing.assert_allclose(filt.P_t_next, oracle_filter.P_t_next)
        assert filt.datetime == filtered_time
        if model_name == 'mc':
            np.testing.assert_allclose(model.depth, oracle.depth)
        if with_troute:
            # The scheduler advances the clock after filtering external flow.
            oracle.datetime += oracle.timedelta
        assert (
            simulation.datetime
            == oracle.datetime
            == start + step * model.timedelta
        )

if __name__ == '__main__':
    test_reservoir_lateral_nan_interpolation(reservoir_data)