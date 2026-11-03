"""
BMI wrapper for KF code base.

This class exposes Muskingum and KF model through BMI.

Written and tested by: Sonam Lama (slama@ua.edu)

"""


import asyncio
import threading
import itertools

import numpy as np
import pandas as pd

from tx_fast_hydrology.simulation import AsyncSimulation
from tx_fast_hydrology.da import KalmanFilter
from tx_fast_hydrology.muskingum import ModelCollection
from tx_fast_hydrology.simulation import AsyncSimulation


#we will have remove this later. This adds streamflow in the upstream boundary
UPSTREAM_INPUTS = {
    "5779305": "5781161",
    "5785187": "5785899",
    "5786029": "5785351",
    }


class BmiKF:
    """BMI wrapper for the TX fast hydrology model collection with DA."""

    _INPUT_VAR_NAMES = ("k", "x", "initial_discharge")
    _OUTPUT_VAR_NAMES = ("reach_list", "discharge")

    def load_config(self, config_path="KF.yaml"):
        config = {}
        with open(config_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip().strip('"')
        return config

    def load_and_build_inputs(self, config):
        model_collection = ModelCollection.from_file(config["network_file_path"])
        forcing = pd.read_csv(config["forcing_file_path"], index_col=0)
        streamflow = pd.read_csv(config["streamflow_file_path"], index_col=0)
        measurements = pd.read_csv(config["measurement_file_path"], index_col=0)

        forcing.index = pd.to_datetime(forcing.index)
        streamflow.index = pd.to_datetime(streamflow.index)
        measurements.index = pd.to_datetime(measurements.index)

        forcing.columns = forcing.columns.astype(str)
        streamflow.columns = streamflow.columns.astype(str)
        measurements.columns = measurements.columns.astype(str)

        input_columns = list(
            itertools.chain.from_iterable(
                [model.reach_ids for model in model_collection.models.values()]
            )
        )
        inputs = pd.DataFrame(0.0, index=forcing.index.copy(), columns=input_columns)

        for col in inputs.columns:
            if col in forcing.columns:
                inputs[col] = forcing[col]

        dt = model_collection.timedelta.seconds
        inputs = inputs.resample(f"{dt}s").mean()
        inputs = inputs.interpolate().bfill().ffill()
        assert not inputs.isnull().any().any()

        for downstream_reach, upstream_reach in UPSTREAM_INPUTS.items():
            inputs[downstream_reach] += streamflow[upstream_reach]

        usgs_to_comid = pd.read_csv(config["usgs_to_comid_file_path"], index_col=0)
        usgs_to_comid["gage_id"] = usgs_to_comid["gage_id"].astype(str)
        usgs_to_comid["comid"] = usgs_to_comid["comid"].astype(str)
        usgs_to_comid = usgs_to_comid[usgs_to_comid["comid"].isin(input_columns)]
        usgs_to_comid = pd.Series(
            usgs_to_comid["comid"].values,
            index=usgs_to_comid["gage_id"].values,
        )

        measurements = measurements[usgs_to_comid.index]
        measurements.columns = measurements.columns.map(usgs_to_comid)
        measurements = measurements.loc[forcing.index[0] : forcing.index[-1]]
        measurements = measurements.dropna(axis=1)
        measurements = measurements.loc[:, ~(measurements == 0.0).all(axis=0).values]
        measurements = measurements.loc[:, ~measurements.columns.duplicated()].copy()
        measurements = measurements.resample(f"{dt}s").mean().interpolate().ffill().bfill()

        return model_collection, measurements, inputs, dt

    def prepare_model(self, model_collection, measurements, inputs, dt):
        for model in model_collection.models.values():
            model_sites = [
                reach_id for reach_id in model.reach_ids if reach_id in measurements.columns
            ]
            if model_sites:
                basin_measurements = measurements[model_sites]
                Q_cov = 1.0 * np.eye(model.n)
                R_cov = 1.0 * np.eye(basin_measurements.shape[1])
                P_t_init = Q_cov.copy()
                kf = KalmanFilter(model, basin_measurements, Q_cov, R_cov, P_t_init)
                model.bind_callback(kf, key="kf")

        for model in model_collection.models.values():
            outlet = model.startnodes[model.startnodes == model.endnodes].item()
            model.set_transmissive_boundary(outlet)

        timedelta = pd.to_timedelta(dt, unit="s")
        for model_name, model in model_collection.models.items():
            model.datetime = inputs.index[0] - timedelta

        return model_collection

    def initialize(self, config_file="KF.yaml"):
        self.config_file = config_file
        self.config = self.load_config(config_file)

        (
            self.model_collection,
            self.measurements,
            self.inputs,
            self.dt,
        ) = self.load_and_build_inputs(self.config)
        self.model_collection = self.prepare_model(
            self.model_collection,
            self.measurements,
            self.inputs,
            self.dt,
        )
        self.simulation = AsyncSimulation(self.model_collection, self.inputs)
        self.outputs_da = None
        self.input_var_store = {name: None for name in self._INPUT_VAR_NAMES}
        self.output_var_store = {name: None for name in self._OUTPUT_VAR_NAMES}

        self._start_datetime = self.model_collection.datetime
        self._start_time = 0.0
        self._current_time = self._start_time
        self._time_step = float(self.model_collection.timedelta.total_seconds())
        self._end_time = float("inf")
        self._initialized = True

    def update(self):
        self.update_until(time_window=self.model_collection.timedelta.total_seconds())

    def update_until(self, time_window):
        self._require_initialized()
        model_timestep = self.model_collection.timedelta.total_seconds()
        if time_window % model_timestep != 0:
            raise ValueError(
                "time_window must be expressible as a multiple of "
                "model_collection.timedelta."
            )

        n_steps = int(time_window / model_timestep)
        for _ in range(n_steps):
            outputs_da = self._run_async(
                self.simulation.simulate(only_one_time_step=True)
            )
            self.outputs_da = pd.concat(
                [series for series in outputs_da.values()],
                axis=1,
            )
            self.output_var_store["reach_list"] = list(self.outputs_da.columns)
            self.output_var_store["discharge"] = self.outputs_da.loc[
                self.model_collection.datetime
            ].values

        self._current_time += time_window

    def finalize(self):
        self._require_initialized()
        self._initialized = False

    def get_component_name(self):
        return "TX Fast Hydrology BMI"

    def get_input_item_count(self):
        return len(self.input_var_store)

    def get_output_item_count(self):
        return len(self.output_var_store)

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_current_time(self):
        return self._current_time

    def get_time_step(self):
        return self._time_step

    def get_time_units(self):
        return "s"

    def get_input_var_names(self):
        return tuple(self.input_var_store.keys())

    def get_output_var_names(self):
        return tuple(self.output_var_store.keys())

    def get_var_type(self, name):
        self._validate_name(name)
        return type(self.input_var_store[name])

    def get_var_units(self, name):
        return NotImplementedError

    def get_var_itemsize(self, name):
        return NotImplementedError

    def get_var_nbytes(self, name):
        return NotImplementedError

    def get_var_location(self, name):
        return NotImplementedError

    def get_var_grid(self, name):
        return NotImplementedError

    def get_grid_rank(self, grid):
        return NotImplementedError

    def get_grid_size(self, grid):
        return NotImplementedError

    def get_grid_type(self, grid):
        return NotImplementedError

    def get_grid_shape(self, grid, shape):
        return NotImplementedError

    def get_grid_spacing(self, grid, spacing):
        return NotImplementedError

    def get_grid_origin(self, grid, origin):
        return NotImplementedError

    def get_grid_node_count(self, grid):
        return NotImplementedError

    def get_grid_edge_count(self, grid):
        return NotImplementedError

    def get_grid_face_count(self, grid):
        return NotImplementedError

    def get_grid_x(self, grid, x):
        return NotImplementedError

    def get_grid_y(self, grid, y):
        return NotImplementedError

    def get_grid_z(self, grid, z):
        return NotImplementedError

    def get_grid_edge_nodes(self, grid, edge_nodes):
        return NotImplementedError

    def get_grid_face_edges(self, grid, face_edges):
        return NotImplementedError

    def get_grid_face_nodes(self, grid, face_nodes):
        return NotImplementedError

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        return NotImplementedError

    def get_value(self, name, dest):
        self._validate_output_name(name)
        dest[:] = self.output_var_store[name]
        return dest

    def get_value_ptr(self, name):
        self._validate_output_name(name)
        return self.output_var_store[name]

    def get_value_at_indices(self, name, dest, inds):
        return NotImplementedError

    def set_value(self, name, src):
        self._validate_input_name(name)
        self.input_var_store[name] = src

    def set_value_at_indices(self, name, inds, src):
        return NotImplementedError


    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:
                error["value"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if error:
            raise error["value"]
        return result["value"]

    def _require_initialized(self):
        if not getattr(self, "_initialized", False):
            raise RuntimeError("BmiKF has not been initialized.")

    def _validate_name(self, name):
        if name not in self.input_var_store and name not in self.output_var_store:
            raise KeyError(f"Unknown BMI variable: {name}")

    def _validate_input_name(self, name):
        if name not in self.input_var_store:
            raise KeyError(f"Unknown BMI input variable: {name}")

    def _validate_output_name(self, name):
        if name not in self.output_var_store:
            raise KeyError(f"Unknown BMI output variable: {name}")


BmiTxFastHydrology = BmiKF