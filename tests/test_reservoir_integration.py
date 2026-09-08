"""
Checks continuity with the reservoir functionality and confirms
quashing of two outstanding bugs related to reservoirs which 
were also terminals as well as initialization with nonzero elements
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tx_fast_hydrology.da import ExtendedKalmanFilter, ReservoirNudging
from tx_fast_hydrology.muskingum import (
    Connection, ModelCollection, Muskingum, Reservoir,
)
from tx_fast_hydrology.simulation import AsyncSimulation, CheckPoint


def reservoir_data(name='reservoir'):
    return {
        'name': name,
        'datetime': pd.Timestamp('2023-05-13T00:00:00Z'),
        'timedelta': pd.Timedelta(hours=1),
        'reach_ids': ['lake-inlet', 'lake-outlet'],
        'reservoir_id': ['lake-1'],
        'outlet_index': 1,
        'A_s': np.array([1.0e6], dtype=np.float64),
        'C_w': np.array([1.6], dtype=np.float64),
        'L': np.array([20.0], dtype=np.float64),
        'L_d': np.array([5.0], dtype=np.float64),
        'h_max': np.array([5.0], dtype=np.float64),
        'h_w': np.array([3.0], dtype=np.float64),
        'h_o': np.array([1.0], dtype=np.float64),
        'C_o': np.array([0.6], dtype=np.float64),
        'O_a': np.array([2.0], dtype=np.float64),
        'h_t': np.array([2.0], dtype=np.float64),
        'o_t': np.array([0.0, 4.0], dtype=np.float64),
    }


def linear_data(name='downstream', reach_id='channel'):
    return {
        'name': name,
        'datetime': pd.Timestamp('2023-05-13T00:00:00Z'),
        'timedelta': pd.Timedelta(hours=1),
        'reach_ids': [reach_id],
        'startnodes': np.array([0], dtype=np.int64),
        'endnodes': np.array([0], dtype=np.int64),
        'K': np.array([3600.0], dtype=np.float64),
        'X': np.array([0.3], dtype=np.float64),
        'o_t': np.array([0.0], dtype=np.float64),
    }


class ReservoirIntegrationTests(unittest.TestCase):
    def test_outlet_rating_curves_cover_orifice_weir_and_dam_regimes(self):
        reservoir = Reservoir(reservoir_data())

        below_weir = np.array([0.5])
        np.testing.assert_allclose(
            reservoir.Q_o(below_weir),
            0.6 * 2.0 * np.sqrt(2.0 * 9.81 * below_weir),
        )
        np.testing.assert_allclose(reservoir.Q_w(below_weir), 0.0)
        np.testing.assert_allclose(reservoir.Q_d(below_weir), 0.0)

        above_weir = np.array([3.0])
        np.testing.assert_allclose(
            reservoir.Q_w(above_weir), 1.6 * 20.0 * 1.0 ** 1.5
        )
        np.testing.assert_allclose(reservoir.Q_d(above_weir), 0.0)

        above_dam = np.array([5.0])
        np.testing.assert_allclose(
            reservoir.Q_w(above_dam), 1.6 * 20.0 * 2.0 ** 1.5
        )
        np.testing.assert_allclose(
            reservoir.Q_d(above_dam),
            1.6 * 20.0 * 5.0 * 1.0 ** 1.5,
        )

    def test_one_step_satisfies_explicit_mass_balance(self):
        reservoir = Reservoir(reservoir_data())
        initial_stage = reservoir.h_t_next.copy()
        forcing = np.array([2.0, 0.0], dtype=np.float64)
        expected_release = float(np.sum(
            reservoir.Q_o(initial_stage)
            + reservoir.Q_w(initial_stage)
            + reservoir.Q_d(initial_stage)
        ))

        reservoir.step(forcing)

        storage_change = float(
            reservoir.A_s[0]
            * (reservoir.h_t_next[0] - initial_stage[0])
        )
        expected_change = reservoir.dt * (
            forcing.sum() - expected_release
        )
        np.testing.assert_allclose(storage_change, expected_change)
        np.testing.assert_allclose(
            reservoir.o_t_next[reservoir.outlet_index], expected_release
        )

    def test_saved_stage_and_outflow_initialize_model_state(self):
        reservoir = Reservoir(reservoir_data())

        np.testing.assert_allclose(reservoir.h_t_next, [2.0])
        np.testing.assert_allclose(reservoir.o_t_next, [0.0, 4.0])

        reservoir.step(np.array([2.0, 0.0], dtype=np.float64))
        self.assertGreater(reservoir.o_t_next[1], 0.0)
        self.assertTrue(np.isfinite(reservoir.h_t_next).all())

    def test_reservoir_state_initializer_validates_and_copies(self):
        reservoir = Reservoir(reservoir_data())
        flow = np.array([1.0, 2.0], dtype=np.float64)
        stage = np.array([2.5], dtype=np.float64)
        reservoir.init_states(o_t_next=flow, h_t_next=stage)
        flow[:] = -1.0
        stage[:] = -1.0

        np.testing.assert_allclose(reservoir.o_t_next, [1.0, 2.0])
        np.testing.assert_allclose(reservoir.h_t_next, [2.5])

    def test_model_collection_round_trip_retains_reservoir_type_and_connection(self):
        reservoir = Reservoir(reservoir_data())
        downstream = Muskingum(linear_data())
        connection = Connection(reservoir, downstream, 1, 0, name='outlet')
        reservoir.sinks.append(connection)
        downstream.sources.append(connection)
        collection = ModelCollection([reservoir, downstream], name='test')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'collection.json'
            collection.dump_model_collection(path)
            restored = ModelCollection.from_file(path)

        restored_reservoir = restored.models['reservoir']
        self.assertIsInstance(restored_reservoir, Reservoir)
        np.testing.assert_allclose(restored_reservoir.h_t_next, [2.0])
        np.testing.assert_allclose(restored_reservoir.o_t_next, [0.0, 4.0])
        self.assertEqual(len(restored_reservoir.sinks), 1)
        self.assertEqual(
            restored_reservoir.sinks[0].downstream_model.name,
            'downstream',
        )

    def test_async_connections_align_only_routed_timestamps(self):
        upstream = Muskingum(linear_data('upstream', 'upstream-channel'))
        reservoir = Reservoir(reservoir_data())
        downstream = Muskingum(
            linear_data('downstream', 'downstream-channel')
        )
        into_reservoir = Connection(
            upstream, reservoir, 0, 0, name='into-reservoir'
        )
        out_of_reservoir = Connection(
            reservoir, downstream, 1, 0, name='out-of-reservoir'
        )
        reservoir_outlet = Connection(
            reservoir, reservoir, 1, 1, name='reservoir-self-loop'
        )
        upstream.sinks.append(into_reservoir)
        reservoir.sources.append(into_reservoir)
        reservoir.sinks.append(out_of_reservoir)
        reservoir.sinks.append(reservoir_outlet)
        reservoir.sources.append(reservoir_outlet)
        downstream.sources.append(out_of_reservoir)
        collection = ModelCollection([upstream, reservoir, downstream])

        times = pd.date_range(
            upstream.datetime, periods=3, freq='h', tz='UTC'
        )
        forcing = pd.DataFrame(
            {
                'upstream-channel': [0.0, 1.0, 1.0],
                'lake-inlet': [0.0, 0.0, 0.0],
                'lake-outlet': [0.0, 0.0, 0.0],
                'downstream-channel': [0.0, 0.0, 0.0],
            },
            index=times,
        )
        outputs = asyncio.run(AsyncSimulation(collection, forcing).simulate())

        self.assertEqual(set(outputs), set(collection.models))
        for frame in outputs.values():
            self.assertEqual(len(frame), len(forcing))
            self.assertTrue(np.isfinite(frame.to_numpy()).all())

    def test_channel_ekf_rejects_reservoir_without_transition_operator(self):
        reservoir = Reservoir(reservoir_data())
        measurements = pd.DataFrame(
            {'lake-outlet': [4.0]}, index=[reservoir.datetime]
        )

        with self.assertRaises(TypeError):
            ExtendedKalmanFilter(
                reservoir,
                measurements,
                Q_cov=np.eye(reservoir.n),
                R_cov=np.eye(1),
                P_t_init=np.eye(reservoir.n),
            )

    def test_rfc_nudging_interpolates_and_replaces_reservoir_outflow(self):
        reservoir = Reservoir(reservoir_data())
        measurements = pd.DataFrame(
            {'lake-outlet': [10.0, 30.0]},
            index=pd.date_range(
                reservoir.datetime, periods=2, freq='2h', tz='UTC'
            ),
        )
        nudging = ReservoirNudging(reservoir, measurements)

        nudging.filter()
        np.testing.assert_allclose(reservoir.o_t_next, [0.0, 10.0])

        reservoir.datetime += pd.Timedelta(hours=1)
        reservoir.o_t_next[reservoir.outlet_index] = -1.0
        nudging.__on_step_end__()
        np.testing.assert_allclose(reservoir.o_t_next, [0.0, 20.0])

    def test_rfc_nudging_stops_after_forecast_and_restores_time(self):
        reservoir = Reservoir(reservoir_data())
        measurements = pd.DataFrame(
            {'lake-outlet': [10.0, 20.0]},
            index=pd.date_range(
                reservoir.datetime, periods=2, freq='h', tz='UTC'
            ),
        )
        nudging = ReservoirNudging(reservoir, measurements)
        checkpoint = CheckPoint(reservoir, timedelta=3600.0)
        reservoir.bind_callback(checkpoint, key='checkpoint')
        reservoir.bind_callback(nudging, key='rfc')

        reservoir.datetime += pd.Timedelta(hours=1)
        nudging.filter()
        reservoir.save_state()
        saved_datetime = nudging.datetime

        reservoir.datetime += pd.Timedelta(hours=1)
        reservoir.o_t_next[reservoir.outlet_index] = 7.0
        nudging.__on_step_end__()
        self.assertEqual(reservoir.o_t_next[reservoir.outlet_index], 7.0)

        nudging.datetime += pd.Timedelta(hours=5)
        reservoir.load_state()
        self.assertEqual(nudging.datetime, saved_datetime)


if __name__ == '__main__':
    unittest.main()
