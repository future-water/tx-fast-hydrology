import uuid
import copy
import logging
import json
from scipy.sparse import lil_matrix, csgraph
import numpy as np
import pandas as pd
import datetime
from tx_fast_hydrology.nutils import interpolate_sample, _ax_bu
from tx_fast_hydrology.simulation import ModelCollection
from tx_fast_hydrology.callbacks import BaseCallback
from tx_fast_hydrology.io import load_model_file, dump_model_file, load_nhd_geojson
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

MIN_SLOPE = 1e-8

DEFAULT_START_TIME = pd.to_datetime(0., utc=True)
DEFAULT_TIMEDELTA = pd.to_timedelta(3600, unit='s')

class Muskingum:
    def __init__(self, data, load_optional=True, create_state_space=False, sparse=False):
        self.sparse = sparse
        self.callbacks = {}
        self.saved_states = {}
        # Read json input file
        if isinstance(data, dict):
            self.load_model(data, load_optional=load_optional)
        elif isinstance(data, str):
            self.load_model_file(data, load_optional=load_optional)
        else:
            raise TypeError('`data` must be a file path or dictionary.')
        # Create logger
        self.logger = logging.getLogger(self.name)
        # Create arrays
        n = self.n
        self.o_t_prev = np.zeros(n, dtype=np.float64)
        self.i_t_next = np.zeros(n, dtype=np.float64)
        self.i_t_prev = np.zeros(n, dtype=np.float64)
        self.init_states(o_t_next=self.o_t_next)
        self.o_t_prev[:] = self.o_t_next[:]
        self.i_t_prev[:] = self.i_t_next[:]
        # Initialize state-space matrices
        if sparse:
            self.A = lil_matrix((n, n))
            self.B = lil_matrix((n, n))
        else:
            self.A = np.zeros((n, n), dtype=np.float64)
            self.B = np.zeros((n, n), dtype=np.float64)
        # Compute alpha, beta, and gamma coefficients
        self.alpha = np.zeros(n, dtype=np.float64)
        self.beta = np.zeros(n, dtype=np.float64)
        self.chi = np.zeros(n, dtype=np.float64)
        self.gamma = np.zeros(n, dtype=np.float64)
        self.compute_muskingum_coeffs()
        if create_state_space:
            self.create_state_space()
        self.save_state()

    @property
    def info(self):
        info_dict = {
            'name' : self.name,
            'datetime' : self.datetime,
            'timedelta' : self.timedelta,
            'reach_ids' : self.reach_ids,
            'startnodes' : self.startnodes,
            'endnodes' : self.endnodes,
            'K' : self.K,
            'X' : self.X,
            'o_t' : self.o_t_next,
            'dx' : self.dx,
            'paths' : self.paths
        }
        return info_dict

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            self.logger.warning(f'`name` is not a string, converting to {new_name}')
            new_name = str(new_name)
        self._name = new_name

    @property
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, new_datetime):
        try:
            assert isinstance(new_datetime, pd.Timestamp)
        except:
            raise TypeError('New datetime must be of type `pd.Timestamp`')
        try:
            assert new_datetime.tz == datetime.timezone.utc
        except:
            raise ValueError('New datetime must be UTC.')
        self._datetime = new_datetime

    @property
    def timedelta(self):
        return self._timedelta

    @timedelta.setter
    def timedelta(self, new_timedelta):
        try:
            assert isinstance(new_timedelta, pd.Timedelta)
        except:
            raise TypeError('New timedelta must be of type `pd.Timedelta`')
        self._timedelta = new_timedelta

    @property
    def o_t(self):
        return self.o_t_next

    @o_t.setter
    def o_t(self, new_o_t):
        try:
            new_o_t = np.asarray(new_o_t, dtype=np.float64)
        except:
            raise TypeError('New `o_t` must be convertible to float64 np.ndarray')
        self.o_t_next = new_o_t

    @property
    def dt(self):
        dt = float(self.timedelta.seconds)
        return dt

    def load_model(self, obj, load_optional=True):
        required_fields = {'name', 'datetime', 'timedelta', 'reach_ids',
                           'startnodes', 'endnodes', 'K', 'X', 'o_t'}
        optional_fields = {'paths', 'dx'}
        defaults = {'name' : str(uuid.uuid4()), 
                    'datetime' : DEFAULT_START_TIME,
                    'timedelta' : DEFAULT_TIMEDELTA,
                    'dx' : None,
                    'paths' : []}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
        try:
            assert isinstance(obj['startnodes'], np.ndarray)
            assert isinstance(obj['endnodes'], np.ndarray)
            assert isinstance(obj['K'], np.ndarray)
            assert isinstance(obj['X'], np.ndarray)
            assert isinstance(obj['o_t'], np.ndarray)
            assert obj['startnodes'].dtype == np.int64
            assert obj['endnodes'].dtype == np.int64
            assert obj['K'].dtype == np.float64
            assert obj['X'].dtype == np.float64
            assert obj['o_t'].dtype == np.float64
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            assert (obj['startnodes'].size == obj['endnodes'].size 
                    == obj['K'].size == obj['X'].size == obj['o_t'].size)
        except:
            raise ValueError('Arrays are not the same length')
        # If optional fields are desired, add to the set of fields
        if load_optional:
            fields = required_fields.union(optional_fields)
        else:
            fields = required_fields
        # Iterate through fields and add as attributes to class instance
        for field in fields:
            if field in defaults:
                default_value = defaults[field]
                value = obj.setdefault(field, default_value)
            else:
                value = obj[field]
            setattr(self, field, value)
        # Add derived attributes
        startnodes, endnodes = self.startnodes, self.endnodes
        self.n = startnodes.size
        self.indegree = self.compute_indegree(startnodes, endnodes)

    def load_model_file(self, file_path, load_optional=True):
        obj = load_model_file(file_path, load_optional=load_optional)
        self.load_model(obj)
    
    def dump_model_file(self, file_path, dump_optional=True):
        obj = self.info
        return dump_model_file(obj, file_path, dump_optional=dump_optional)

    @classmethod
    def from_nhd_geojson(cls, data, **kwargs):
        parsed_data = load_nhd_geojson(data)
        newinstance = cls(parsed_data, **kwargs)
        return newinstance

    def compute_indegree(self, startnodes, endnodes):
        self_loops = []
        for i in range(len(startnodes)):
            if endnodes[i] == startnodes[i]:
                self_loops.append(i)
        indegree = np.bincount(endnodes.ravel(), minlength=startnodes.size)
        for self_loop in self_loops:
            indegree[self_loop] -= 1
        return indegree

    def compute_alpha(self, K, X, dt):
        # TODO: Correct order in numerator?
        alpha = (dt - 2 * K * X) / (2 * K * (1 - X) + dt)
        return alpha

    def compute_beta(self, K, X, dt):
        beta = (dt + 2 * K * X) / (2 * K * (1 - X) + dt)
        return beta

    def compute_chi(self, K, X, dt):
        chi = (2 * K * (1 - X) - dt) / (2 * K * (1 - X) + dt)
        return chi

    def compute_gamma(self, K, X, dt):
        gamma = dt / (K * (1 - X) + dt / 2)
        return gamma

    def compute_muskingum_coeffs(self, K=None, X=None, dt=None):
        self.logger.info('Computing Muskingum coefficients...')
        if K is None:
            K = self.K
        if X is None:
            X = self.X
        if dt is None:
            dt = self.dt
        self.alpha[:] = self.compute_alpha(K, X, dt)
        self.beta[:] = self.compute_beta(K, X, dt)
        self.chi[:] = self.compute_chi(K, X, dt)
        self.gamma[:] = self.compute_gamma(K, X, dt)

    def create_state_space(self, overwrite_old=True):
        self.logger.info('Creating state-space system...')
        startnodes = self.startnodes
        endnodes = self.endnodes
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        indegree = self.indegree
        A = self.A
        B = self.B
        if overwrite_old:
            self.A.fill(0.0)
            self.B.fill(0.0)
        startnodes = startnodes[(indegree == 0)]
        A, B = self.muskingum_matrix(
            A, B, startnodes, endnodes, alpha, beta, chi, gamma, indegree
        )
        self.A = A
        self.B = B

    def muskingum_matrix(
        self, A, B, startnodes, endnodes, alpha, beta, chi, gamma, indegree
    ):
        m = startnodes.size
        n = endnodes.size
        indegree_t = indegree.copy()
        for k in range(m):
            startnode = startnodes[k]
            endnode = endnodes[startnode]
            while indegree_t[startnode] == 0:
                alpha_i = alpha[startnode]
                beta_i = beta[startnode]
                chi_i = chi[startnode]
                gamma_i = gamma[startnode]
                alpha_j = alpha[endnode]
                beta_j = beta[endnode]
                A[startnode, startnode] = chi_i
                B[startnode, startnode] = gamma_i
                if startnode != endnode:
                    A[endnode, startnode] += beta_j
                    A[endnode] += alpha_j * A[startnode]
                    B[endnode] += alpha_j * B[startnode]
                indegree_t[endnode] -= 1
                startnode = endnode
                endnode = endnodes[startnode]
        return A, B

    def init_states(self, o_t_next=None, i_t_next=None):
        if o_t_next is None:
            self.o_t_next[:] = 0.
        else:
            self.o_t_next[:] = o_t_next[:]
        if i_t_next is None:
            self.i_t_next[:] = 0.
            np.add.at(self.i_t_next, self.endnodes, self.o_t_next[self.startnodes])
        else:
            self.i_t_next[:] = i_t_next[:]

    def step(self, p_t_next, num_iter=1, inc_t=False):
        raise NotImplementedError
        A = self.A
        B = self.B
        dt = self.dt
        timedelta = self.timedelta
        o_t_prev = self.o_t_next
        o_t_next = A @ o_t_prev + B @ p_t_next
        self.o_t_next = o_t_next
        self.o_t_prev = o_t_prev
        if inc_t:
            self.t += dt
            self.datetime += self.timedelta

    def step_iter(self, p_t_next, timedelta=None, inc_t=True):
        if timedelta is None:
            timedelta = self.timedelta
            dt = self.dt
        else:
            dt = float(timedelta.seconds)
        startnodes = self.startnodes
        endnodes = self.endnodes
        indegree = self.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        o_t_prev = self.o_t_next
        i_t_prev = self.i_t_next
        if dt != self.dt:
            self.logger.warning('Timestep has changed. Recomputing Muskingum coefficients.')
            self.compute_muskingum_coeffs(dt=dt)
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        for _, callback in self.callbacks.items():
            callback.__on_step_start__()
        i_t_next, o_t_next = _ax_bu(sub_startnodes, endnodes, alpha, beta, chi, gamma,
                                    i_t_prev, o_t_prev, p_t_next, indegree)
        self.o_t_next = o_t_next
        self.o_t_prev = o_t_prev
        self.i_t_next = i_t_next
        self.i_t_prev = i_t_prev
        self.datetime += timedelta
        for _, callback in self.callbacks.items():
            callback.__on_step_end__()
        self.logger.debug(f'Stepped to time {self.datetime}')

    def simulate(self, dataframe, **kwargs):
        raise NotImplementedError
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        # assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        dataframe = dataframe[self.reach_ids]
        dataframe.index = dataframe.index.tz_convert("UTC")
        self.datetime = dataframe.index[0]
        self.t = self.datetime.timestamp()
        for index in dataframe.index:
            p_t_next = dataframe.loc[index, :].values
            self.step(p_t_next, **kwargs)
            yield self

    def simulate_iter(self, dataframe, start_time=None, end_time=None, o_t_init=None, **kwargs):
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        # Set start and end times
        # TODO: Put in checks here
        if end_time is None:
            end_time = dataframe.index.max()
        else:
            try:
                assert isinstance(end_time, pd.Timestamp)
            except:
                raise TypeError('`end_time` must be of type `pd.Timestamp`')
        if start_time is None:
            start_time = self.datetime
        else:
            self.datetime = start_time
            try:
                assert o_t_init is not None
            except:
                ValueError('If `start_time` is specified, initial state `o_t_init` must be provided.')
        if o_t_init is not None:
            self.init_states(o_t_next=o_t_init)
        # Execute pre-simulation callbacks
        for _, callback in self.callbacks.items():
            callback.__on_simulation_start__()
        # Crop input data to model reaches
        dataframe = dataframe[self.reach_ids]
        while self.datetime < end_time:
            next_timestep = self.datetime + self.timedelta
            p_t_next = interpolate_sample(float(next_timestep.value), 
                                          dataframe.index.astype(int).astype(float).values,
                                          dataframe.values) 
            self.step_iter(p_t_next, **kwargs)
            yield self
        # Execute post-simulation callbacks
        for _, callback in self.callbacks.items():
            callback.__on_simulation_end__()

    def save_state(self):
        self.logger.info(f'Saving state for model {self.name} at time {self.datetime}...')
        self.saved_states["datetime"] = self.datetime
        # TODO: Don't need to store `i_t_next`
        self.saved_states["i_t_next"] = self.i_t_next.copy()
        self.saved_states["o_t_next"] = self.o_t_next.copy()
        for _, callback in self.callbacks.items():
            callback.__on_save_state__()

    def load_state(self):
        self.datetime = self.saved_states["datetime"]
        self.i_t_next = self.saved_states["i_t_next"]
        self.o_t_next = self.saved_states["o_t_next"]
        self.logger.info(f'Loading state for model {self.name} at time {self.datetime}...')
        for _, callback in self.callbacks.items():
            callback.__on_load_state__()

    def plot(self, ax, *args, **kwargs):
        paths = self.paths
        for path in paths:
            for subpath in path:
                ax.plot(subpath[:,0], subpath[:,1], *args, **kwargs)
    
    def bind_callback(self, callback, key='callback'):
        assert isinstance(callback, BaseCallback)
        self.callbacks[key] = callback

    def unbind_callback(self, key):
        return self.callbacks.pop(key)

    def copy(self):
        return copy.deepcopy(self)

    def split(self, indices, create_state_space=True):
        self = copy.deepcopy(self)
        startnode_indices = np.asarray([np.flatnonzero(self.startnodes == i).item()
                                        for i in indices])
        endnode_indices = self.endnodes[startnode_indices]
        # Cut watershed at indices
        self.endnodes[startnode_indices] = indices
        # Find connected components
        adj = lil_matrix((self.n, self.n), dtype=int)
        for i, j in zip(self.startnodes, self.endnodes):
            adj[j, i] = 1
        n_components, labels = csgraph.connected_components(adj)
        index_map = pd.Series(np.arange(self.n), index=self.startnodes)
        outer_startnodes = labels[startnode_indices].astype(int).tolist()
        outer_endnodes = labels[endnode_indices].astype(int).tolist()
        # Re-order
        new_outer_startnodes = []
        new_outer_endnodes = []
        new_startnode_indices = []
        new_endnode_indices = []
        for k in range(n_components):
            new_outer_startnodes.append(k)
            if k in outer_startnodes:
                pos = outer_startnodes.index(k)
                new_outer_endnodes.append(outer_endnodes[pos])
                new_startnode_indices.append(startnode_indices[pos])
                new_endnode_indices.append(endnode_indices[pos])
            else:
                new_outer_endnodes.append(k)
                new_startnode_indices.append(-1)
                new_endnode_indices.append(-1)
        outer_startnodes = np.asarray(new_outer_startnodes, dtype=int)
        outer_endnodes = np.asarray(new_outer_endnodes, dtype=int)
        startnode_indices = np.asarray(new_startnode_indices, dtype=int)
        endnode_indices = np.asarray(new_endnode_indices, dtype=int)
        outer_indegree = np.bincount(outer_endnodes, minlength=outer_endnodes.size)
        # Create sub-watershed models
        components = {}
        for component in range(n_components):
            components[component] = {}
            selection = (labels == component)
            sub_startnodes = self.startnodes[selection]
            sub_endnodes = self.endnodes[selection]
            new_startnodes = np.arange(len(sub_startnodes))
            node_map = pd.Series(new_startnodes, index=sub_startnodes)
            new_endnodes = node_map.loc[sub_endnodes].values
            sub_index_map = index_map[sub_startnodes].values
            # Create sub-watershed model
            sub_model = copy.deepcopy(self)
            sub_model.name = str(component)
            sub_model.startnodes = new_startnodes
            sub_model.endnodes = new_endnodes
            sub_model.n = len(sub_model.startnodes)
            sub_model.indegree = self.indegree[sub_index_map]
            sub_model.reach_ids = np.asarray(self.reach_ids)[sub_index_map].tolist()
            sub_model.dx = self.dx[sub_index_map]
            sub_model.K = self.K[sub_index_map]
            sub_model.X = self.X[sub_index_map]
            sub_model.alpha = self.alpha[sub_index_map]
            sub_model.beta = self.beta[sub_index_map]
            sub_model.chi = self.chi[sub_index_map]
            sub_model.gamma = self.gamma[sub_index_map]
            sub_model.A = np.zeros((sub_model.n, sub_model.n))
            sub_model.B = np.zeros((sub_model.n, sub_model.n))
            sub_model.o_t_prev = self.o_t_prev[sub_index_map]
            sub_model.i_t_prev = self.i_t_prev[sub_index_map]
            sub_model.o_t_next = self.o_t_next[sub_index_map]
            sub_model.i_t_next = self.i_t_next[sub_index_map]
            sub_model.paths = [self.paths[i] for i in sub_index_map]
            if create_state_space:
                sub_model.create_state_space()
            sub_model.save_state()
            components[component]['model'] = sub_model
            components[component]['node_map'] = node_map
        # Create connections between sub-watersheds
        for component in range(n_components):
            startnode = outer_startnodes[component]
            endnode = outer_endnodes[component]
            upstream_node_map = components[startnode]['node_map']
            downstream_node_map = components[endnode]['node_map']
            startnode_index = startnode_indices[component]
            endnode_index = endnode_indices[component]
            components[startnode]['terminal_node'] = startnode_index
            components[startnode]['entry_node'] = endnode_index
            if (startnode_index >= 0) and (endnode_index >= 0):
                upstream_model = components[startnode]['model']
                downstream_model = components[endnode]['model']
                exit_index = upstream_node_map[startnode_index]
                entry_index = downstream_node_map[endnode_index]
                downstream_model.indegree[entry_index] -= 1
                downstream_model.i_t_next[entry_index] -= upstream_model.o_t_next[exit_index]
        model_collection = ModelCollection(components, outer_startnodes,
                                           outer_endnodes, startnode_indices,
                                           endnode_indices)
        return model_collection

def _solve_normal_depth(h, Q, B, z, mann_n, So):
    Q_computed = _Q(h, Q, B, z, mann_n, So)
    return Q_computed - Q


def _Q(h, Q, B, z, mann_n, So):
    A = h * (B + z * h)
    P = B + 2 * h * np.sqrt(1 + z**2)
    Q = (np.sqrt(So) / mann_n) * A ** (5 / 3) / P ** (2 / 3)
    return Q

def _dQ_dh(h, Q, B, z, mann_n, So):
    num_0 = 5 * np.sqrt(So) * (h * (B + h * z)) ** (2 / 3) * (B + 2 * h * z)
    den_0 = 3 * mann_n * (B + 2 * h * np.sqrt(z + 1)) ** (2 / 3)
    num_1 = 4 * np.sqrt(So) * np.sqrt(z + 1) * (h * (B + h * z)) ** (5 / 3)
    den_1 = 3 * mann_n * (B + 2 * h * np.sqrt(z + 1)) ** (5 / 3)
    t0 = num_0 / den_0
    t1 = num_1 / den_1
    return t0 - t1
