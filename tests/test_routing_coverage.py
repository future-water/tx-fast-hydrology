"""Routing reference calculations, time handling, and restart behavior."""

import numpy as np
import pandas as pd
import pytest

from tx_fast_hydrology.muskingum import Muskingum
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge
from tx_fast_hydrology.simulation import CheckPoint


def test_linear_recurrence_and_confluence(linear):
    previous = linear.o_t_next.copy()
    inflow = linear.i_t_next.copy()
    forcing = np.array([0.2, 0.1, 0.3])
    # Explicit three-reach recurrence, independently of the production traversal.
    expected = (
        linear.beta * inflow + linear.chi * previous + linear.gamma * forcing
    )
    expected[2] += linear.alpha[2] * expected[:2].sum()
    linear.step(forcing)
    np.testing.assert_allclose(linear.o_t_next, expected)
    np.testing.assert_allclose(linear.i_t_next, [0.0, 0.0, expected[:2].sum()])
    np.testing.assert_allclose(linear.alpha + linear.beta + linear.chi, 1.0)


def test_initial_linear_routing_matches_state_space(linear):
    previous = linear.o_t_next.copy()
    forcing = np.array([0.2, 0.1, 0.3])
    linear.step(forcing)
    np.testing.assert_allclose(
        linear.o_t_next, linear.A @ previous + linear.B @ forcing
    )


@pytest.mark.parametrize(
    'ends,flows,inflow',
    [
        ([0], [3.0], [0.0]),
        ([1, 2, 2], [2.0, 1.0, 3.0], [0.0, 2.0, 1.0]),
        ([1, 1, 3, 3], [2.0, 1.0, 3.0, 4.0], [0.0, 2.0, 0.0, 3.0]),
        ([0, 1, 2], [2.0, 1.0, 3.0], [0.0, 0.0, 0.0]),
    ],
)
def test_initial_inflows_for_outlet_topologies(
    channel_data, ends, flows, inflow
):
    size = len(ends)
    data = dict(
        channel_data,
        reach_ids=[str(i) for i in range(size)],
        startnodes=np.arange(size, dtype=np.int64),
        endnodes=np.array(ends, dtype=np.int64),
        K=np.full(size, 300.0),
        X=np.full(size, 0.3),
        o_t=np.array(flows),
        dx=np.full(size, 1000.0),
        paths=[],
    )
    model = Muskingum(data, create_state_space=True)
    # Outlets keep their self-edges in topology but add no inflow to themselves.
    np.testing.assert_array_equal(model.i_t_next, inflow)
    forcing = np.full(size, 0.1)
    expected = model.A @ model.o_t_next + model.B @ forcing
    model.step(forcing)
    np.testing.assert_allclose(model.o_t_next, expected)
    # An explicit boundary inflow must take precedence over reconstruction.
    prescribed = np.arange(size, dtype=float) + 10.0
    model.init_states(o_t_next=np.array(flows), i_t_next=prescribed)
    np.testing.assert_array_equal(model.i_t_next, prescribed)


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
def test_simulation_interpolates_forcing_and_reorders_columns(
    channel_data, geometry, kind
):
    kwargs = {'geometry': geometry} if kind is MuskingumCunge else {}
    model, oracle = (
        kind(channel_data.copy(), **kwargs),
        kind(channel_data.copy(), **kwargs),
    )
    start = model.datetime
    inputs = pd.DataFrame(
        {'3': [0.0, 0.4], '1': [0.2, 0.6], '2': [0.1, 0.3]},
        index=[start, start + pd.Timedelta(minutes=10)],
    )
    actual = [state.o_t_next.copy() for state in model.simulate(inputs)]
    expected = []
    for forcing in ([0.4, 0.2, 0.2], [0.6, 0.3, 0.4]):
        oracle.step(np.array(forcing))
        expected.append(oracle.o_t_next.copy())
    np.testing.assert_allclose(actual, expected)
    assert model.datetime == inputs.index[-1]


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
def test_missing_forcing_column_is_rejected(channel_data, geometry, kind):
    model = kind(
        channel_data,
        **({'geometry': geometry} if kind is MuskingumCunge else {}),
    )
    inputs = pd.DataFrame({'1': [1.0], '2': [2.0]}, index=[model.datetime])
    with pytest.raises(AssertionError):
        list(model.simulate(inputs))


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
def test_missing_lateral_forcing_is_reported(channel_data, geometry, kind):
    model = kind(
        channel_data,
        **({'geometry': geometry} if kind is MuskingumCunge else {}),
    )
    inputs = pd.DataFrame(
        [[np.nan, 2.0, 0.0], [np.nan, 2.0, 0.0]],
        index=[model.datetime, model.datetime + model.timedelta],
        columns=model.reach_ids,
    )
    with pytest.raises(ValueError, match='finite.*1'):
        list(model.simulate(inputs))


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
@pytest.mark.parametrize('unit', ['ns', 'us'])
@pytest.mark.parametrize('single_step', [False, True])
def test_lateral_nan_gaps_match_prescribed_routing(
    channel_data, geometry, kind, unit, single_step
):
    kwargs = {'geometry': geometry} if kind is MuskingumCunge else {}
    model, oracle = (
        kind(channel_data.copy(), **kwargs),
        kind(channel_data.copy(), **kwargs),
    )
    start = model.datetime
    # Independent gaps, unequal timestamp spacing, and missing endpoints.
    inputs = pd.DataFrame(
        {
            '3': [0.0, np.nan, 0.2, np.nan],
            '1': [0.2, np.nan, np.nan, 0.8],
            '2': [np.nan, 0.1, np.nan, 0.6],
        },
        index=(start + pd.to_timedelta([0, 5, 20, 30], unit='min')).as_unit(
            unit
        ),
    )
    original = inputs.copy(deep=True)
    expected = []
    for forcing in (
        [0.3, 0.1, 0.05],
        [0.4, 0.2, 0.1],
        [0.5, 0.3, 0.15],
        [0.6, 0.4, 0.2],
        [0.7, 0.5, 0.2],
        [0.8, 0.6, 0.2],
    ):
        oracle.step(np.array(forcing))
        expected.append(oracle.o_t_next.copy())
    if single_step:
        actual = []
        for _ in expected:
            assert (
                model.simulate_iter(inputs, only_one_time_step=True) is model
            )
            actual.append(model.o_t_next.copy())
    else:
        actual = [state.o_t_next.copy() for state in model.simulate(inputs)]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert model.datetime == start + pd.Timedelta(minutes=30)
    pd.testing.assert_frame_equal(inputs, original)


def test_empty_lateral_forcing_is_rejected(linear):
    inputs = pd.DataFrame(
        columns=linear.reach_ids,
        index=pd.DatetimeIndex([], tz='UTC'),
        dtype=float,
    )
    with pytest.raises(ValueError, match='at least one timestamp'):
        list(linear.simulate(inputs))


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
def test_checkpoint_replays_routing(channel_data, geometry, kind):
    model = kind(
        channel_data,
        **({'geometry': geometry} if kind is MuskingumCunge else {}),
    )
    model.step(np.array([0.2, 0.1, 0.0]))
    model.save_state()
    model.step(np.array([0.3, 0.4, 0.0]))
    expected = model.o_t_next.copy()
    # Replay the same forcing from the checkpoint and compare routed flow.
    model.load_state()
    model.step(np.array([0.3, 0.4, 0.0]))
    np.testing.assert_allclose(model.o_t_next, expected)


def test_callbacks_and_periodic_checkpoint(linear):
    checkpoint = CheckPoint(linear, timedelta=300)
    linear.bind_callback(checkpoint)
    inputs = pd.DataFrame(
        0.0,
        index=pd.date_range(linear.datetime, periods=3, freq='5min'),
        columns=linear.reach_ids,
    )
    list(
        linear.simulate(
            inputs, start_time=linear.datetime, o_t_init=np.ones(3)
        )
    )
    assert checkpoint.model_saved
    saved_time = inputs.index[1]
    linear.load_state()
    assert linear.datetime == saved_time
    assert linear.unbind_callback('callback') is checkpoint
    with pytest.raises(ValueError):
        CheckPoint(linear)
    fixed = CheckPoint(linear, checkpoint_time=saved_time)
    fixed.__on_simulation_start__()
    fixed.set_checkpoint(saved_time + linear.timedelta)
    assert not fixed.model_saved


def test_transmissive_boundary_and_copy(linear):
    copied = linear.copy()
    copied.set_transmissive_boundary(2)
    copied.step(np.zeros(3))
    assert copied.o_t_next[2] == pytest.approx(copied.o_t_next[:2].sum())
    np.testing.assert_array_equal(linear.o_t_next, [2.0, 1.0, 3.0])


def test_datetime_requires_timezone(linear):
    with pytest.raises(ValueError):
        linear.datetime = pd.Timestamp('2023-01-01')


def test_explicit_initial_flows(linear):
    linear.init_states()
    np.testing.assert_array_equal(linear.o_t, np.zeros(3))
    linear.init_states(o_t_next=np.ones(3), i_t_next=np.array([0.0, 0.0, 2.0]))
    np.testing.assert_array_equal(linear.i_t_next, [0.0, 0.0, 2.0])
    linear.o_t = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(linear.o_t, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    'mutation,error',
    [
        (lambda data: data.pop('K'), ValueError),
        (lambda data: data.update(K=np.ones(2)), ValueError),
    ],
)
def test_model_validation(channel_data, mutation, error):
    mutation(channel_data)
    with pytest.raises(error):
        Muskingum(channel_data)


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
def test_explicit_shorter_step_updates_coefficients(
    channel_data, geometry, kind
):
    model = kind(
        channel_data,
        **({'geometry': geometry} if kind is MuskingumCunge else {}),
    )
    start = model.datetime
    model.step(np.array([0.2, 0.1, 0.0]), timedelta=pd.Timedelta(seconds=60))
    assert model.datetime == start + pd.Timedelta(seconds=60)
    np.testing.assert_allclose(
        model.chi,
        (2 * model.K * (1 - model.X) - 60)
        / (2 * model.K * (1 - model.X) + 60),
    )


def test_unset_mc_geometry_is_rejected(channel_data):
    with pytest.raises(ValueError, match='not set'):
        MuskingumCunge(channel_data).step(np.zeros(3))


def test_mc_saved_depth_shape_validation(channel_data):
    with pytest.raises(ValueError, match='Saved depth'):
        MuskingumCunge(dict(channel_data, depth=[0.0]))


def test_mc_geometry_checks(channel_data, geometry):
    with pytest.raises(ValueError, match='missing required'):
        MuskingumCunge(channel_data, geometry={})
    geometry['Bw'] = [1.0, 2.0]
    with pytest.raises(ValueError, match='length'):
        MuskingumCunge(channel_data, geometry=geometry)
