"""Small deterministic fixtures; no network, archives, or sibling checkouts."""

from contextlib import closing
import sqlite3

import numpy as np
import pandas as pd
import pytest

from tx_fast_hydrology.muskingum import Muskingum
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge


@pytest.fixture
def channel_data():
    # Two headwaters feed reach 3; its self-edge marks the outlet.
    return {
        'name': 'three-reach',
        'datetime': pd.Timestamp('2023-05-13', tz='UTC'),
        'timedelta': pd.Timedelta(minutes=5),
        'reach_ids': ['1', '2', '3'],
        'startnodes': np.arange(3, dtype=np.int64),
        'endnodes': np.array([2, 2, 2], dtype=np.int64),
        'K': np.full(3, 300.0),
        'X': np.full(3, 0.3),
        'o_t': np.array([2.0, 1.0, 3.0]),
        'dx': np.full(3, 1000.0),
        'paths': [[[[0.0, 0.0], [1.0, 1.0]]]] * 3,
    }


@pytest.fixture
def geometry():
    return {
        key: np.full(3, value)
        for key, value in {
            'So': 0.001,
            'dx': 1000.0,
            'n': 0.035,
            'Cs': 1.0,
            'Bw': 5.0,
            'Tw': 15.0,
            'TwCC': 30.0,
            'nCC': 0.06,
        }.items()
    }


@pytest.fixture
def linear(channel_data):
    return Muskingum(channel_data, create_state_space=True)


@pytest.fixture
def mc(channel_data, geometry):
    return MuskingumCunge(channel_data, geometry=geometry)


@pytest.fixture
def fabric_frames():
    flowpaths = pd.DataFrame(
        {'id': ['1', '2', '3'], 'toid': ['a', 'a', 'out']}
    )
    # Shuffle hydraulic rows to test the join by flowpath ID.
    attributes = pd.DataFrame(
        {
            'id': ['3', '1', '2'],
            'Length_m': [1200.0, 800.0, 900.0],
            'So': [0.001, 0.0012, 0.0008],
            'n': [0.035, 0.033, 0.034],
            'ChSlp': [0.5] * 3,
            'BtmWdth': [12.0, 8.0, 9.0],
            'TopWdth': [18.0, 12.0, 13.0],
            'TopWdthCC': [40.0, 30.0, 32.0],
            'nCC': [0.08] * 3,
        }
    )
    nexus = pd.DataFrame({'id': ['a', 'out'], 'toid': ['3', None]})
    return flowpaths, attributes, nexus


@pytest.fixture
def gpkg(tmp_path, fabric_frames):
    # The SQLite loader only needs these non-geometric tables.
    path = tmp_path / 'three.gpkg'
    with closing(sqlite3.connect(path)) as connection:
        for name, frame in zip(
            ('flowpaths', 'flowpath-attributes', 'nexus'), fabric_frames
        ):
            frame.to_sql(name, connection, index=False)
    return path
