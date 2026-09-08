from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from tx_fast_hydrology.hydrofabric import (
    HydrofabricNetwork, load_hydrofabric,
)
from tx_fast_hydrology.muskingum import ModelCollection
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge


def fixture_tables():
    flowpaths = pd.DataFrame({
        'id': ['wb-1', 'wb-2', 'wb-3'],
        'toid': ['nex-1', 'nex-1', 'nex-out'],
    })
    attributes = pd.DataFrame({
        'id': ['wb-3', 'wb-1', 'wb-2'],
        'Length_m': [1200.0, 800.0, 900.0],
        'So': [0.0010, 0.0012, 0.0008],
        'n': [0.035, 0.033, 0.034],
        'ChSlp': [0.5, 0.5, 0.5],
        'BtmWdth': [12.0, 8.0, 9.0],
        'TopWdth': [18.0, 12.0, 13.0],
        'TopWdthCC': [40.0, 30.0, 32.0],
        'nCC': [0.08, 0.08, 0.08],
    })
    nexus = pd.DataFrame({
        'id': ['nex-1', 'nex-out'],
        'toid': ['wb-3', None],
    })
    return flowpaths, attributes, nexus


class HydrofabricIntegrationTests(unittest.TestCase):
    def test_topology_and_geometry_align_by_flowpath_id(self):
        flowpaths, attributes, nexus = fixture_tables()
        network = HydrofabricNetwork.from_frames(
            flowpaths, attributes, nexus=nexus
        )

        self.assertEqual(network.reach_ids, ('wb-1', 'wb-2', 'wb-3'))
        np.testing.assert_array_equal(network.endnodes, [2, 2, 2])
        np.testing.assert_allclose(network.dx, [800.0, 900.0, 1200.0])
        self.assertEqual(network.outlets, ('wb-3',))

    def test_selected_subnetwork_converts_cut_edge_to_local_outlet(self):
        flowpaths, attributes, nexus = fixture_tables()
        network = HydrofabricNetwork.from_frames(
            flowpaths,
            attributes,
            nexus=nexus,
            reach_ids=['wb-1'],
        )

        self.assertEqual(network.reach_ids, ('wb-1',))
        np.testing.assert_array_equal(network.endnodes, [0])
        self.assertEqual(network.outlets, ('wb-1',))

    def test_geopackage_constructs_main_muskingum_cunge_model(self):
        flowpaths, attributes, nexus = fixture_tables()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.gpkg'
            with closing(sqlite3.connect(path)) as connection:
                flowpaths.to_sql('flowpaths', connection, index=False)
                attributes.to_sql(
                    'flowpath-attributes', connection, index=False
                )
                nexus.to_sql('nexus', connection, index=False)

            network = load_hydrofabric(path)
            model = MuskingumCunge.from_hydrofabric(
                path, timedelta='5min',
                initial_flow={'wb-3': 0.0, 'wb-2': 0.0, 'wb-1': 0.0},
            )

        self.assertEqual(model.reach_ids, list(network.reach_ids))
        np.testing.assert_array_equal(model.endnodes, network.endnodes)
        np.testing.assert_allclose(model.mann_n, network.n)
        model.step(np.array([0.1, 0.2, 0.0], dtype=np.float64))
        self.assertTrue(np.isfinite(model.o_t_next).all())

    def test_muskingum_cunge_collection_round_trip_retains_geometry(self):
        flowpaths, attributes, nexus = fixture_tables()
        network = HydrofabricNetwork.from_frames(
            flowpaths, attributes, nexus=nexus
        )
        model = MuskingumCunge.from_hydrofabric(
            network, name='mc', timedelta='5min'
        )
        model.depth[:] = [0.1, 0.2, 0.3]
        collection = ModelCollection([model])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'collection.json'
            collection.dump_model_collection(path)
            restored = ModelCollection.from_file(path)

        restored_model = restored.models['mc']
        self.assertIsInstance(restored_model, MuskingumCunge)
        np.testing.assert_allclose(restored_model.dx, network.dx)
        np.testing.assert_allclose(restored_model.mann_n, network.n)
        np.testing.assert_allclose(restored_model.depth, [0.1, 0.2, 0.3])


if __name__ == '__main__':
    unittest.main()
