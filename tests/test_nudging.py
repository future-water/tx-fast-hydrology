"""Focused unit tests for NWM streamflow nudging."""

import math

import numpy as np
import pandas as pd

from tx_fast_hydrology.da import (
    StreamflowNudging,
    TRouteStreamflowNudging,
    WRFHydroStreamflowNudging,
)
from tx_fast_hydrology.mc_kernel_troute import _mc_ax_bu
from tx_fast_hydrology.muskingum import Muskingum
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge


START = pd.Timestamp('2023-05-13T00:00:00Z')


def build_model(reaches=1):
    endnodes = np.arange(reaches, dtype=np.int64)
    if reaches > 1:
        endnodes[:-1] = np.arange(1, reaches, dtype=np.int64)
    data = {
        'name': 'nudging-test',
        'datetime': START,
        'timedelta': pd.Timedelta(minutes=5),
        'reach_ids': [str(index + 1) for index in range(reaches)],
        'startnodes': np.arange(reaches, dtype=np.int64),
        'endnodes': endnodes,
        'K': np.full(reaches, 300.0),
        'X': np.full(reaches, 0.3),
        'o_t': np.full(reaches, 10.0),
        'dx': np.full(reaches, 1000.0),
    }
    geometry = {
        'So': np.full(reaches, 0.001),
        'dx': np.full(reaches, 1000.0),
        'n': np.full(reaches, 0.035),
        'Cs': np.full(reaches, 1.0),
        'Bw': np.full(reaches, 5.0),
        'Tw': np.full(reaches, 10.0),
        'TwCC': np.full(reaches, 20.0),
        'nCC': np.full(reaches, 0.07),
    }
    return MuskingumCunge(data, geometry=geometry)


def build_linear_model(reaches=1):
    endnodes = np.arange(reaches, dtype=np.int64)
    if reaches > 1:
        endnodes[:-1] = np.arange(1, reaches, dtype=np.int64)
    data = {
        'name': 'linear-nudging-test',
        'datetime': START,
        'timedelta': pd.Timedelta(minutes=5),
        'reach_ids': [str(index + 1) for index in range(reaches)],
        'startnodes': np.arange(reaches, dtype=np.int64),
        'endnodes': endnodes,
        'K': np.full(reaches, 300.0, dtype=np.float64),
        'X': np.full(reaches, 0.3, dtype=np.float64),
        'o_t': np.linspace(10.0, 10.0 + reaches - 1, reaches),
        'dx': np.full(reaches, 1000.0, dtype=np.float64),
    }
    return Muskingum(data)


def test_wrf_temporal_weights_and_quality_scale_the_innovation():
    model = build_model()
    model.datetime = START + pd.Timedelta(minutes=7, seconds=30)
    measurements = pd.DataFrame(
        {'1': [20.0, 40.0]},
        index=[START, START + pd.Timedelta(minutes=15)],
    )
    quality = pd.DataFrame(
        {'1': [0.5, 0.5]}, index=measurements.index,
    )
    nudging = WRFHydroStreamflowNudging(
        model, measurements, quality=quality,
        temporal_persistence=False,
    )

    nudge = nudging.update()

    # Equal temporal weights average innovations of 10 and 30, then quality
    # scales that average by 0.5.
    np.testing.assert_allclose(nudge, [10.0])
    np.testing.assert_allclose(model.o_t_next, [20.0])


def test_wrf_forecast_persistence_uses_exponential_coefficient():
    model = build_model()
    model.datetime = START + pd.Timedelta(minutes=120)
    measurements = pd.DataFrame({'1': [20.0]}, index=[START])
    nudging = WRFHydroStreamflowNudging(
        model, measurements, tau_minutes=15,
        exp_coefficients=120.0, temporal_persistence=True,
        assimilation_end=START,
    )

    nudge = nudging.update()

    np.testing.assert_allclose(nudge, [10.0 / math.e])


def test_wrf_previous_nudge_enters_both_mc_upstream_terms():
    model = build_model(reaches=2)
    measurements = pd.DataFrame({'2': [20.0]}, index=[START])
    nudging = WRFHydroStreamflowNudging(
        model, measurements, tau_minutes=1,
        temporal_persistence=False,
    )
    model.bind_callback(nudging, key='streamflow_nudging')
    nudging.update()

    previous_flow = model.o_t_next.copy()
    previous_nudge = nudging.previous_nudge.copy()
    expected = _mc_ax_bu(
        model.startnodes[model.indegree == 0], model.endnodes,
        model.indegree, previous_flow, np.zeros(model.n), model.depth.copy(),
        300.0, model.So, model.dx, model.mann_n, model.Cs, model.Bw,
        model.Tw, model.TwCC, model.nCC, previous_nudge,
        model.assume_short_ts,
    )[0]

    model.step(np.zeros(model.n), timedelta=pd.Timedelta(minutes=5))

    np.testing.assert_allclose(model.o_t_next, expected, rtol=0, atol=1e-12)
    assert nudging.previous_nudge[1] == 0.0


def test_wrf_previous_nudge_enters_both_linear_upstream_terms():
    model = build_linear_model()
    measurements = pd.DataFrame({'1': [20.0]}, index=[START])
    nudging = WRFHydroStreamflowNudging(
        model, measurements, tau_minutes=1,
        temporal_persistence=False,
    )
    model.bind_callback(nudging, key='streamflow_nudging')
    nudging.update()

    nudge = float(nudging.previous_nudge[0])
    expected = (
        model.alpha[0] * nudge
        + model.beta[0] * (model.i_t_next[0] + nudge)
        + model.chi[0] * model.o_t_next[0]
    )

    model.step(np.zeros(model.n, dtype=np.float64))

    np.testing.assert_allclose(model.o_t_next, [expected], rtol=0, atol=1e-12)


def test_wrf_zero_nudge_preserves_linear_routing_exactly():
    baseline = build_linear_model(reaches=3)
    nudged = build_linear_model(reaches=3)
    measurements = pd.DataFrame({'1': [np.nan]}, index=[START])
    callback = WRFHydroStreamflowNudging(
        nudged, measurements, temporal_persistence=False,
    )
    nudged.bind_callback(callback, key='streamflow_nudging')

    lateral_inflow = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    baseline.step(lateral_inflow)
    nudged.step(lateral_inflow)

    np.testing.assert_array_equal(nudged.o_t_next, baseline.o_t_next)
    np.testing.assert_array_equal(nudged.i_t_next, baseline.i_t_next)


def test_save_and_load_restore_forecast_nudge_state():
    model = build_model()
    measurements = pd.DataFrame({'1': [20.0]}, index=[START])
    nudging = StreamflowNudging(model, measurements)
    model.bind_callback(nudging, key='streamflow_nudging')
    nudging.update()
    model.save_state()
    saved = nudging.previous_nudge.copy()

    nudging.previous_nudge[:] = 999.0
    model.load_state()

    np.testing.assert_array_equal(nudging.previous_nudge, saved)


def test_default_callback_is_troute_and_directly_inserts_observation():
    model = build_model()
    measurements = pd.DataFrame({'1': [25.0]}, index=[START])

    nudging = StreamflowNudging(model, measurements)
    nudge = nudging.update()

    assert isinstance(nudging, TRouteStreamflowNudging)
    np.testing.assert_allclose(nudge, [15.0])
    np.testing.assert_allclose(model.o_t_next, [25.0])


def test_troute_nudging_directly_inserts_into_linear_model():
    model = build_linear_model()
    measurements = pd.DataFrame({'1': [25.0]}, index=[START])

    nudge = StreamflowNudging(model, measurements).update()

    np.testing.assert_allclose(nudge, [15.0])
    np.testing.assert_allclose(model.o_t_next, [25.0])


def test_troute_linear_model_routes_from_corrected_state_next_step():
    model = build_linear_model()
    expected_model = build_linear_model()
    measurements = pd.DataFrame({'1': [25.0]}, index=[START])
    nudging = StreamflowNudging(
        model, measurements, temporal_persistence=False,
    )
    model.bind_callback(nudging, key='streamflow_nudging')
    nudging.update()

    # Reproduce the state t-route passes into the next routing step: corrected
    # discharge, with all other linear-model states unchanged.
    expected_model.o_t_next[:] = 25.0
    model.step(np.zeros(model.n, dtype=np.float64))
    expected_model.step(np.zeros(expected_model.n, dtype=np.float64))

    np.testing.assert_allclose(model.o_t_next, expected_model.o_t_next)


def test_troute_decay_matches_simple_da_equation():
    model = build_model()
    measurements = pd.DataFrame({'1': [20.0]}, index=[START])
    nudging = StreamflowNudging(
        model, measurements, decay_coefficient=120.0,
    )
    nudging.update()

    model.datetime = START + pd.Timedelta(minutes=120)
    model.o_t_next[:] = 8.0
    nudge = nudging.update()

    # simple_da.pyx: (last observation - current model) * exp(-age/a)
    np.testing.assert_allclose(nudge, [(20.0 - 8.0) / math.e])
    np.testing.assert_allclose(model.o_t_next, [8.0 + 12.0 / math.e])


def test_troute_interpolates_observations_to_routing_timestep():
    model = build_model()
    model.datetime = START + pd.Timedelta(minutes=5)
    measurements = pd.DataFrame(
        {'1': [20.0, 50.0]},
        index=[START, START + pd.Timedelta(minutes=15)],
    )

    nudge = StreamflowNudging(model, measurements).update()

    # t-route preprocesses 15-minute time slices onto its routing timestep.
    np.testing.assert_allclose(nudge, [20.0])
    np.testing.assert_allclose(model.o_t_next, [30.0])


def test_troute_quality_is_a_gate_not_a_nudge_multiplier():
    measurements = pd.DataFrame({'1': [20.0]}, index=[START])

    rejected_model = build_model()
    rejected = StreamflowNudging(
        rejected_model, measurements, quality=0.5,
        quality_threshold=1.0,
    ).update()
    np.testing.assert_array_equal(rejected, [0.0])

    accepted_model = build_model()
    accepted = StreamflowNudging(
        accepted_model, measurements, quality=0.5,
        quality_threshold=0.5,
    ).update()
    # Once accepted, the observation replaces the model value in full.
    np.testing.assert_allclose(accepted, [10.0])
    np.testing.assert_allclose(accepted_model.o_t_next, [20.0])


def test_troute_does_not_add_wrf_upstream_nudge_term():
    model = build_model(reaches=2)
    measurements = pd.DataFrame({'2': [20.0]}, index=[START])
    nudging = StreamflowNudging(
        model, measurements, temporal_persistence=False,
    )
    model.bind_callback(nudging, key='streamflow_nudging')
    nudging.update()

    previous_flow = model.o_t_next.copy()
    zeros = np.zeros(model.n, dtype=np.float64)
    expected = _mc_ax_bu(
        model.startnodes[model.indegree == 0], model.endnodes,
        model.indegree, previous_flow, zeros, model.depth.copy(),
        300.0, model.So, model.dx, model.mann_n, model.Cs, model.Bw,
        model.Tw, model.TwCC, model.nCC, zeros,
        model.assume_short_ts,
    )[0]

    model.step(zeros, timedelta=pd.Timedelta(minutes=5))

    np.testing.assert_allclose(model.o_t_next, expected, rtol=0, atol=1e-12)


def test_troute_missing_observation_without_history_is_passthrough():
    model = build_model()
    measurements = pd.DataFrame(
        {'1': [np.nan]}, index=[START + pd.Timedelta(minutes=5)],
    )

    nudge = StreamflowNudging(model, measurements).update()

    np.testing.assert_array_equal(nudge, [0.0])
    np.testing.assert_array_equal(model.o_t_next, [10.0])
