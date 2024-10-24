import uuid
import copy
import logging
from scipy.sparse import lil_matrix, csgraph
from scipy.optimize import root_scalar
import numpy as np
import pandas as pd
import datetime
from tx_fast_hydrology.nutils import interpolate_sample, _ax_bu
from tx_fast_hydrology.simulation import ModelCollection
from tx_fast_hydrology.callbacks import BaseCallback
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

logger = logging.getLogger(__name__)

MIN_SLOPE = 1e-8

DEFAULT_START_TIME = pd.to_datetime(0., utc=True)
DEFAULT_TIMEDELTA = pd.to_timedelta(3600, unit='s')

class Muskingum:
    def __init__(self, json, init_o_t=None, dt=3600.0, t_0=0.0, name=None,
                 create_state_space=True, sparse=False, verbose=False):
        self.logger = logger
        self.verbose = verbose
        self.sparse = sparse
        self.callbacks = {}
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            assert isinstance(name, str)
            self.name = name
        # Read json input file
        assert isinstance(json, dict)
        self.read_nhd_file(json)
        # Construct network topology
        self.construct_network()
        # Create arrays
        n = self.endnodes.size
        self.n = n
        self.alpha = np.zeros(n, dtype=np.float64)
        self.beta = np.zeros(n, dtype=np.float64)
        self.chi = np.zeros(n, dtype=np.float64)
        self.gamma = np.zeros(n, dtype=np.float64)
        self.h = np.zeros(n, dtype=np.float64)
        self.Qn = np.zeros(n, dtype=np.float64)
        self.K = np.zeros(n, dtype=np.float64)
        self.X = np.zeros(n, dtype=np.float64)
        self.o_t_next = np.zeros(n, dtype=np.float64)
        self.o_t_prev = np.zeros(n, dtype=np.float64)
        self.i_t_next = np.zeros(n, dtype=np.float64)
        self.i_t_prev = np.zeros(n, dtype=np.float64)
        if init_o_t is None:
            self.o_t_next[:] = 1e-3 * np.ones(n, dtype=np.float64)
            self.init_states(o_t_next=self.o_t_next)
            self.o_t_prev[:] = self.o_t_next[:]
            self.i_t_prev = self.i_t_next[:]
        else:
            assert isinstance(init_o_t, np.ndarray)
            assert init_o_t.ndim == 1
            assert init_o_t.size == n
            self.o_t_next[:] = init_o_t.copy()
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
        # Compute parameters
        timedelta = datetime.timedelta(seconds=dt)
        self.t = t_0
        self.datetime = pd.to_datetime(t_0, unit="s", utc=True)
        self.timedelta = timedelta
        self.saved_states = {}
        # TODO: Should allow these to be set
        self.K[:] = 3600.
        self.X[:] = 0.29
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
            'o_t' : self.o_t_next
        }
        return info_dict

    @property
    def dt(self):
        dt = float(self.timedelta.seconds)
        return dt

    def read_nhd_file(self, d):
        logger.info('Reading NHD file...')
        node_ids = [i['attributes']['COMID'] for i in d['features']]
        link_ids = [i['attributes']['COMID'] for i in d['features']]
        paths = [np.asarray(i['geometry']['paths']) for i in d['features']]
        reach_ids = link_ids
        reach_ids = [str(x) for x in reach_ids]
        source_node_ids = [i['attributes']['COMID'] for i in d['features']]
        target_node_ids = [i['attributes']['toCOMID'] for i in d['features']]
        dx = np.asarray([i['attributes']['Shape_Length'] for i in d['features']]) #* 1000 # km to m
        self.reach_ids = reach_ids
        self.node_ids = node_ids
        self.source_node_ids = source_node_ids
        self.target_node_ids = target_node_ids
        self.dx = dx
        self.paths = paths

    def read_hydraulic_geometry(self, d):
        raise NotImplementedError
        source_node_ids = self.source_node_ids
        target_node_ids = self.target_node_ids
        # Set trapezoidal geometry
        Bws = []
        h_maxes = []
        zs = []
        for link in d["links"]:
            geom = link["hydrologic_routing"]["muskingum_cunge_station"][
                "cross_section"
            ]
            Tw = geom[3]["lr"] - geom[0]["lr"]
            Bw = geom[2]["lr"] - geom[1]["lr"]
            h_max = geom[0]["z"] - geom[1]["z"]
            z = (geom[1]["lr"] - geom[0]["lr"]) / h_max
            Bws.append(Bw)
            h_maxes.append(h_max)
            zs.append(z)
        self.Bw = np.asarray(Bws)
        self.h_max = np.asarray(h_maxes)
        self.z = np.asarray(zs)
        n = self.Bw.size
        node_elevs = {i["uid"]: i["location"]["z"] for i in d["nodes"]}
        So = []
        for i in range(len(source_node_ids)):
            source_node_id = source_node_ids[i]
            target_node_id = target_node_ids[i]
            z_0 = node_elevs[source_node_id]
            z_1 = node_elevs.get(target_node_id, z_0 - 1)
            dx_i = d["links"][i]["length"]
            slope = (z_0 - z_1) / dx_i
            So.append(max(slope, MIN_SLOPE))
        self.So = np.asarray(So)
        self.Ar = np.zeros(n, dtype=np.float64)
        self.P = np.zeros(n, dtype=np.float64)
        self.R = np.zeros(n, dtype=np.float64)
        self.Tw = np.zeros(n, dtype=np.float64)

    def construct_network(self):
        logger.info('Constructing network...')
        # NOTE: node_ids and source_node_ids should always be the same for NHD
        # But not necessarily so for original json files
        node_ids = self.node_ids
        source_node_ids = self.source_node_ids
        target_node_ids = self.target_node_ids
        self_loops = []
        node_index_map = pd.Series(np.arange(len(node_ids)), index=node_ids)
        startnodes = node_index_map.reindex(source_node_ids, fill_value=-1).values
        endnodes = node_index_map.reindex(target_node_ids, fill_value=-1).values
        for i in range(len(startnodes)):
            if endnodes[i] == -1:
                self_loops.append(i)
                endnodes[i] = startnodes[i]
        indegree = self.compute_indegree(startnodes, endnodes)
        self.startnodes = startnodes
        self.endnodes = endnodes
        self.indegree = indegree

    def compute_indegree(self, startnodes, endnodes):
        self_loops = []
        for i in range(len(startnodes)):
            if endnodes[i] == startnodes[i]:
                self_loops.append(i)
        indegree = np.bincount(endnodes.ravel(), minlength=startnodes.size)
        for self_loop in self_loops:
            indegree[self_loop] -= 1
        return indegree

    def compute_normal_depth(self, Q, mindepth=0.01):
        So = self.So
        Bw = self.Bw
        z = self.z
        mann_n = self.mann_n
        h = []
        for i in range(Q.size):
            Qi = Q[i]
            Bi = Bw[i]
            zi = z[i]
            Soi = So[i]
            mann_ni = mann_n[i]
            result = root_scalar(
                _solve_normal_depth,
                args=(Qi, Bi, zi, mann_ni, Soi),
                x0=1.0,
                fprime=_dQ_dh,
                method="newton",
            )
            h.append(result.root)
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, mindepth)
        self.h[:] = h

    def compute_hydraulic_geometry(self, h):
        Bw = self.Bw
        z = self.z
        So = self.So
        mann_n = self.mann_n
        Ar = (Bw + h * z) * h
        P = Bw + 2 * h * np.sqrt(1.0 + z**2)
        Tw = Bw + 2 * z * h
        R = Ar / P
        R[P <= 0] = 0.0
        Qn = np.sqrt(So) / mann_n * Ar * R ** (2 / 3)
        self.Ar[:] = Ar
        self.P[:] = P
        self.R[:] = R
        self.Tw[:] = Tw
        self.Qn[:] = Qn

    def compute_K_and_X(self, h, Q):
        logger.info('Computing K and X...')
        Bw = self.Bw
        z = self.z
        R = self.R
        Tw = self.Tw
        So = self.So
        Ar = self.Ar
        mann_n = self.mann_n
        dt = self.dt
        dx = self.dx
        K = self.K
        X = self.X
        # c_k = (np.sqrt(So) / mann_n) * ( (5. / 3.) * R**(2. / 3.) -
        #          ((2. / 3.) * R**(5. / 3.) * (2. * np.sqrt(1. + z**2)
        #                                       / (Bw + 2 * h * z))))
        # c_k = 1.67 * Q / Ar
        # c_k = np.maximum(c_k, 0.)
        # K[:] = dx / c_k
        # cond = c_k > 0
        # K[cond] = np.maximum(dt, dx[cond] / c_k[cond])
        # K[~cond] = dt
        X[:] = 0.29
        # X[cond] = np.minimum(0.5, np.maximum(0.,
        #           0.5 * (1 - (Q[cond] / (2. * Tw[cond] * So[cond]
        #                                  * c_k[cond] * dx[cond])))))
        # X[~cond] = 0.5
        self.K = K
        self.X = X

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
        logger.info('Computing Muskingum coefficients...')
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
        logger.info('Creating state-space system...')
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
            logger.warning('Timestep has changed. Recomputing Muskingum coefficients.')
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
        self.t += dt
        for _, callback in self.callbacks.items():
            callback.__on_step_end__()
        logger.debug(f'Stepped to time {self.datetime}')

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

    def simulate_iter(self, dataframe, **kwargs):
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        # Execute pre-simulation callbacks
        for _, callback in self.callbacks.items():
            callback.__on_simulation_start__()
        # Crop input data to model reaches
        dataframe = dataframe[self.reach_ids]
        timedelta = self.timedelta
        for index in dataframe.index:
            # TODO: Remove. This is too error prone
            # timedelta = index - self.datetime
            p_t_next = interpolate_sample(float(index.value), 
                                          dataframe.index.astype(int).astype(float).values,
                                          dataframe.values) 
            #p_t_next = dataframe.loc[index, :].values
            self.step_iter(p_t_next, timedelta=timedelta, **kwargs)
            yield self
        # Execute post-simulation callbacks
        for _, callback in self.callbacks.items():
            callback.__on_simulation_end__()

    def save_state(self):
        logger.info(f'Saving state for model {self.name} at time {self.datetime}...')
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
        logger.info(f'Loading state for model {self.name} at time {self.datetime}...')
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

    def split(self, indices, inputs, create_state_space=True):
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
            components[component]['input'] = inputs[sub_model.reach_ids].copy()
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
