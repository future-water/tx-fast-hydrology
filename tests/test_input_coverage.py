"""Tiny actual SQLite, NetCDF and JSON files exercise loader boundaries."""

from contextlib import closing
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tx_fast_hydrology.hydrofabric import HydrofabricNetwork, load_hydrofabric
from tx_fast_hydrology.io import ModelEncoder
from tx_fast_hydrology.muskingum import (
    Muskingum,
    ModelCollection,
    load_muskingum_model_file,
)
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge, ROUTELINK_MAP


def test_routelink_netcdf_alignment_and_routing(
    tmp_path, channel_data, geometry
):
    # Public load_routelink accepts a table; xarray reads the small real file.
    order = np.array([2, 0, 1])
    fields = {'link': ('feature_id', np.array([3, 1, 2]))}
    for source, target in ROUTELINK_MAP.items():
        key = 'n' if target == 'mann_n' else target
        values = geometry[key].copy()
        if target == 'dx':
            values = np.array([800.0, 900.0, 1200.0])
        fields[source] = ('feature_id', values[order])
    path = tmp_path / 'RouteLink.nc'
    xr.Dataset(fields).to_netcdf(path, engine='h5netcdf')
    with xr.open_dataset(path, engine='h5netcdf') as dataset:
        table = dataset.to_dataframe().reset_index()
    original = table.copy(deep=True)
    model = MuskingumCunge(channel_data)
    assert model.load_routelink(table) is model
    np.testing.assert_array_equal(model.dx, [800.0, 900.0, 1200.0])
    pd.testing.assert_frame_equal(table, original)
    model.step(np.array([0.1, 0.2, 0.0]))
    assert np.isfinite(model.o_t_next).all()
    with pytest.raises(ValueError, match='not found'):
        model.load_routelink(table.iloc[:1])
    with pytest.raises(ValueError, match='missing required'):
        model.load_routelink(table.drop(columns='n'))
    assert (
        model.load_routelink(
            table.rename(columns={'link': 'comid'}), id_field='comid'
        )
        is model
    )


@pytest.mark.parametrize('kind', [Muskingum, MuskingumCunge])
def test_json_collection_preserves_routing(
    tmp_path, channel_data, geometry, kind
):
    model = kind(
        channel_data,
        **({'geometry': geometry} if kind is MuskingumCunge else {}),
    )
    model.step(np.array([0.1, 0.2, 0.0]))
    path = tmp_path / 'model.json'
    ModelCollection([model]).dump_model_collection(path)
    restored = ModelCollection.from_file(path).models[model.name]
    np.testing.assert_allclose(restored.o_t_next, model.o_t_next)
    assert type(restored) is kind
    if kind is MuskingumCunge:
        model.step(np.array([0.2, 0.3, 0.1]))
        restored.step(np.array([0.2, 0.3, 0.1]))
        np.testing.assert_allclose(restored.o_t_next, model.o_t_next)


def test_linear_json_reload_preserves_next_step(tmp_path, linear):
    linear.step(np.array([0.1, 0.2, 0.0]))
    path = tmp_path / 'collection.json'
    ModelCollection([linear]).dump_model_collection(path)
    restored = ModelCollection.from_file(path).models[linear.name]
    # Check reconstructed inflow before comparing several continuation steps.
    np.testing.assert_allclose(restored.i_t_next, linear.i_t_next)
    for forcing in ([0.2, 0.3, 0.1], [0.0, 0.0, 0.0], [0.4, 0.0, 0.2]):
        for model in (linear, restored):
            model.step(np.array(forcing))
        np.testing.assert_allclose(restored.o_t_next, linear.o_t_next)
        np.testing.assert_allclose(restored.i_t_next, linear.i_t_next)
        assert restored.datetime == linear.datetime


def test_linear_model_json_file(tmp_path, linear):
    path = tmp_path / 'linear.json'
    path.write_text(
        json.dumps(linear.info, cls=ModelEncoder), encoding='utf-8'
    )
    restored = Muskingum.from_model_file(path)
    np.testing.assert_array_equal(restored.endnodes, linear.endnodes)
    restored.load_model_file(path)
    np.testing.assert_array_equal(restored.o_t, linear.o_t)


@pytest.mark.parametrize('load_optional', [False, True])
@pytest.mark.parametrize('path_type', [str, Path])
def test_constructor_reads_json_path(
    tmp_path, linear, load_optional, path_type
):
    path = tmp_path / 'linear.json'
    path.write_text(
        json.dumps(linear.info, cls=ModelEncoder), encoding='utf-8'
    )
    # Exercise the file constructor with both path types and loading flags.
    restored = Muskingum(path_type(path), load_optional=load_optional)
    np.testing.assert_array_equal(restored.o_t, linear.o_t)
    np.testing.assert_array_equal(restored.endnodes, linear.endnodes)
    assert restored.datetime == linear.datetime
    if load_optional:
        np.testing.assert_array_equal(restored.dx, linear.dx)
    else:
        assert restored.dx is None
        assert restored.paths == []


@pytest.mark.parametrize('payload', [{}, {'startnodes': [0], 'endnodes': [0]}])
def test_json_missing_fields(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='must contain'):
        load_muskingum_model_file(path)


def test_json_mismatched_arrays(tmp_path, channel_data):
    channel_data['K'] = np.ones(2)
    path = tmp_path / 'bad.json'
    path.write_text(
        json.dumps(channel_data, cls=ModelEncoder), encoding='utf-8'
    )
    with pytest.raises(ValueError, match='same length'):
        load_muskingum_model_file(path)


def test_json_numpy_scalars():
    result = json.loads(
        json.dumps({'i': np.int64(2), 'f': np.float32(1.5)}, cls=ModelEncoder)
    )
    assert result == {'i': 2, 'f': 1.5}


def test_hydrofabric_geometry_load_and_initial_conditions(gpkg, mc):
    assert mc.load_hydrofabric(gpkg) is mc
    np.testing.assert_allclose(mc.dx, [800.0, 900.0, 1200.0])
    network = load_hydrofabric(gpkg)
    flows = pd.Series([3.0, 2.0, 1.0], index=['3', '2', '1'])
    model = MuskingumCunge.from_hydrofabric(
        network,
        datetime='2023-05-13T05:00:00+05:00',
        initial_flow=flows,
        initial_depth=[0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(model.o_t_next, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(model.depth, [0.1, 0.2, 0.3])
    assert model.datetime == pd.Timestamp('2023-05-13', tz='UTC')
    naive = MuskingumCunge.from_hydrofabric(network, datetime='2023-05-13')
    assert naive.datetime == model.datetime
    with pytest.raises(ValueError, match='already loaded'):
        MuskingumCunge.from_hydrofabric(network, reach_ids=['1'])
    mc.reach_ids = ['3', '2', '1']
    with pytest.raises(ValueError, match='order'):
        mc.load_hydrofabric(gpkg)
    mc.reach_ids = ['1', '2', '3']
    mc.endnodes = np.array([0, 1, 2])
    with pytest.raises(ValueError, match='topology'):
        mc.load_hydrofabric(gpkg)


@pytest.mark.parametrize(
    'values,message',
    [
        ({'1': 1.0}, 'missing'),
        (pd.Series([1.0], index=['1']), 'missing'),
        ([1.0, 2.0], 'shape'),
        ([1.0, np.nan, 2.0], 'non-finite'),
    ],
)
def test_hydrofabric_initial_value_validation(gpkg, values, message):
    with pytest.raises(ValueError, match=message):
        MuskingumCunge.from_hydrofabric(gpkg, initial_flow=values)


@pytest.mark.parametrize(
    'table', ['flowpaths', 'flowpath-attributes', 'nexus']
)
def test_geopackage_missing_tables(gpkg, table):
    # Remove one required table while retaining the other valid fixtures.
    with closing(sqlite3.connect(gpkg)) as connection:
        connection.execute(f'DROP TABLE "{table}"')
        connection.commit()
    with pytest.raises(ValueError, match='missing tables'):
        load_hydrofabric(gpkg)


def test_geopackage_missing_columns_and_bad_paths(gpkg, tmp_path):
    with closing(sqlite3.connect(gpkg)) as connection:
        connection.execute('ALTER TABLE "flowpaths" DROP COLUMN "toid"')
        connection.commit()
    with pytest.raises(ValueError, match='missing required columns'):
        load_hydrofabric(gpkg)
    with pytest.raises(FileNotFoundError):
        load_hydrofabric(tmp_path / 'missing.gpkg')
    bad = tmp_path / 'fabric.csv'
    bad.touch()
    with pytest.raises(ValueError, match='GeoPackage'):
        load_hydrofabric(bad)


@pytest.mark.parametrize(
    'case,message',
    [
        ('duplicate_flowpath', 'Duplicate flowpath'),
        ('duplicate_attributes', 'Duplicate hydraulic'),
        ('conflicting_nexus', 'multiple downstream'),
        ('missing_nexus', 'Missing examples'),
        ('missing_attribute', 'missing/non-numeric'),
        ('non_numeric', 'missing/non-numeric'),
        ('cycle', 'cycle'),
        ('absent_selection', 'absent'),
    ],
)
def test_hydrofabric_bad_topology_and_attributes(fabric_frames, case, message):
    flowpaths, attributes, nexus = fabric_frames
    # Change one part of an otherwise valid fabric for each error case.
    kwargs = {}
    if case == 'duplicate_flowpath':
        flowpaths = pd.concat([flowpaths, flowpaths.iloc[:1]])
    elif case == 'duplicate_attributes':
        attributes = pd.concat([attributes, attributes.iloc[:1]])
    elif case == 'conflicting_nexus':
        nexus = pd.concat([nexus, pd.DataFrame({'id': ['a'], 'toid': ['2']})])
    elif case == 'missing_nexus':
        nexus = nexus.iloc[1:]
    elif case == 'missing_attribute':
        attributes = attributes.iloc[1:]
    elif case == 'non_numeric':
        attributes['n'] = ['invalid', '.03', '.04']
    elif case == 'cycle':
        nexus.loc[1, 'toid'] = '1'
    else:
        kwargs['reach_ids'] = ['absent']
    with pytest.raises(ValueError, match=message):
        HydrofabricNetwork.from_frames(
            flowpaths, attributes, nexus=nexus, **kwargs
        )


@pytest.mark.parametrize(
    'field,value',
    [
        ('Length_m', 0.0),
        ('n', -1.0),
        ('BtmWdth', 0.0),
        ('TopWdth', -1.0),
        ('ChSlp', -1.0),
        ('TopWdthCC', -1.0),
        ('nCC', -1.0),
    ],
)
def test_hydrofabric_invalid_geometry(fabric_frames, field, value):
    flowpaths, attributes, nexus = fabric_frames
    attributes.loc[0, field] = value
    with pytest.raises(ValueError, match='Hydraulic field'):
        HydrofabricNetwork.from_frames(flowpaths, attributes, nexus=nexus)


def test_hydrofabric_zero_slope_and_repeated_identical_nexus(fabric_frames):
    flowpaths, attributes, nexus = fabric_frames
    attributes['So'] = 0.0
    network = HydrofabricNetwork.from_frames(
        flowpaths, attributes, nexus=pd.concat([nexus, nexus.iloc[:1]])
    )
    np.testing.assert_array_equal(network.So, [1e-8] * 3)
