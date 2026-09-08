"""Muskingum-Cunge model."""
import numpy as np
import pandas as pd

from tx_fast_hydrology.hydrofabric import HydrofabricNetwork, load_hydrofabric
from tx_fast_hydrology.muskingum import (
    DEFAULT_START_TIME, DEFAULT_TIMEDELTA, Muskingum,
)
from tx_fast_hydrology.mc_kernel_troute import _mc_ax_bu

# RouteLink field -> model attribute
ROUTELINK_MAP = {
    'So': 'So',
    'n': 'mann_n',
    'nCC': 'nCC',
    'ChSlp': 'Cs',
    'BtmWdth': 'Bw',
    'TopWdth': 'Tw',
    'TopWdthCC': 'TwCC',
    'Length': 'dx',
}

MIN_SLOPE = 1e-8
_GEOM_FIELDS = ('So', 'dx', 'mann_n', 'Cs', 'Bw', 'Tw', 'TwCC', 'nCC')


class MuskingumCunge(Muskingum):
    """Nonlinear Muskingum-Cunge routing with EKF support.

    ``create_state_space`` is accepted for compatibility with the linear ``Muskingum``
    class but ignored because the nonlinear transition changes at each substep.
    """
    def __init__(self, data, geometry=None, load_optional=True,
                 create_state_space=False, sparse=False,
                 assume_short_ts=None, **kwargs):
        """
        Parameters
        ------
        data : dict or str
            ModelCollection submodel dictionary or JSON path.
        geometry : dict
            Per-reach arrays keyed by the fields in `_GEOM_FIELDS`.
        assume_short_ts : bool, optional
            Use previous upstream discharge for both MC upstream-flow terms.
            The default is True, matching the bundled NWM validation data.
            Set False for t-route's upstream-to-downstream current-flow mode.
        """
        saved_depth = None
        if isinstance(data, dict):
            if geometry is None:
                geometry = data.get('geometry')
            saved_depth = data.get('depth')
            if assume_short_ts is None:
                assume_short_ts = data.get('assume_short_ts', True)
        elif assume_short_ts is None:
            assume_short_ts = True
        if not isinstance(assume_short_ts, (bool, np.bool_)):
            raise TypeError('assume_short_ts must be a boolean')
        self.assume_short_ts = bool(assume_short_ts)

        super().__init__(data, load_optional=load_optional,
                         create_state_space=False, sparse=sparse)
        self.model_type = 'muskingum_cunge'
        self.depth = np.zeros(self.n, dtype=np.float64)
        if saved_depth is not None:
            saved_depth = np.asarray(saved_depth, dtype=np.float64)
            if saved_depth.shape != (self.n,):
                raise ValueError('Saved depth must contain one value per reach')
            self.depth[:] = saved_depth
        self.saved_states['depth'] = self.depth.copy()

        self.velocity = np.zeros(self.n, dtype=np.float64)
        # K and X are hydraulic states for Muskingum-Cunge, rather than fixed
        # calibration parameters.
        # Keep both the prior/current values and a
        # timestamped history so data-assimilation callbacks can use the
        # coefficients from the previous substep.
        self.K_t_prev = self.K.copy()
        self.X_t_prev = self.X.copy()
        self.K_history = {self.datetime: self.K.copy()}
        self.X_history = {self.datetime: self.X.copy()}
        self._coefficient_dt = self.dt
        # Geometry may be supplied now or loaded from RouteLink/Hydrofabric.
        for f in _GEOM_FIELDS:
            if not hasattr(self, f):
                setattr(self, f, None)
        if geometry is not None:
            self.set_geometry(geometry)
        # The base constructor saves before the nonlinear hydraulic states
        # exist, so refresh the initial checkpoint now that K/X/depth do.
        self.save_state()

    @property
    def info(self):
        """Return the original model fields plus nonlinear MC state."""
        info = super().info
        if all(getattr(self, field, None) is not None
               for field in _GEOM_FIELDS):
            info['geometry'] = {
                field: getattr(self, field).copy()
                for field in _GEOM_FIELDS
            }
        info['depth'] = self.depth.copy()
        info['assume_short_ts'] = self.assume_short_ts
        return info

    @staticmethod
    def _align_hydrofabric_values(values, network, name, default=0.0):
        if values is None:
            return np.full(network.size, default, dtype=np.float64)
        if isinstance(values, pd.Series):
            keyed = values.copy()
            keyed.index = keyed.index.map(str)
            missing = set(network.reach_ids) - set(keyed.index)
            if missing:
                raise ValueError(f'{name} is missing {len(missing)} flowpaths, '
                                 f'e.g. {sorted(missing)[:5]}')
            array = keyed.reindex(network.reach_ids).to_numpy(np.float64)
        elif isinstance(values, dict):
            keyed = {str(key): value for key, value in values.items()}
            missing = set(network.reach_ids) - set(keyed)
            if missing:
                raise ValueError(f'{name} is missing {len(missing)} flowpaths, '
                                 f'e.g. {sorted(missing)[:5]}')
            array = np.asarray(
                [keyed[reach_id] for reach_id in network.reach_ids],
                dtype=np.float64,
            )
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.shape != (network.size,):
            raise ValueError(f'{name} must have shape ({network.size},), '
                             f'got {array.shape}')
        if not np.isfinite(array).all():
            raise ValueError(f'{name} contains non-finite values')
        return array.copy()

    @classmethod
    def from_hydrofabric(cls, source, reach_ids=None, name='hydrofabric',
                         datetime=None, timedelta=None, initial_flow=None,
                         initial_depth=None, **kwargs):
        """Construct a routing model directly from a NextGen GeoPackage."""
        if isinstance(source, HydrofabricNetwork):
            if reach_ids is not None:
                raise ValueError('reach_ids cannot be used with an already '
                                 'loaded HydrofabricNetwork')
            network = source
        else:
            network = load_hydrofabric(source, reach_ids=reach_ids)

        if datetime is None:
            datetime = DEFAULT_START_TIME
        else:
            datetime = pd.Timestamp(datetime)
            if datetime.tz is None:
                datetime = datetime.tz_localize('UTC')
            else:
                datetime = datetime.tz_convert('UTC')
        if timedelta is None:
            timedelta = DEFAULT_TIMEDELTA
        else:
            timedelta = pd.Timedelta(timedelta)

        flow = cls._align_hydrofabric_values(
            initial_flow, network, 'initial_flow'
        )
        dt = float(timedelta.total_seconds())
        data = {
            'name': str(name),
            'datetime': datetime,
            'timedelta': timedelta,
            'reach_ids': list(network.reach_ids),
            'startnodes': np.arange(network.size, dtype=np.int64),
            'endnodes': network.endnodes.copy(),
            # K/X are replaced by the hydraulic solve at the first step.
            'K': np.full(network.size, dt, dtype=np.float64),
            'X': np.full(network.size, 0.3, dtype=np.float64),
            'o_t': flow,
            'dx': network.dx.copy(),
        }
        geometry = {
            'So': network.So,
            'dx': network.dx,
            'n': network.n,
            'Cs': network.Cs,
            'Bw': network.Bw,
            'Tw': network.Tw,
            'TwCC': network.TwCC,
            'nCC': network.nCC,
        }
        model = cls(data, geometry=geometry, **kwargs)
        model.hydrofabric_network = network
        if initial_depth is not None:
            model.depth[:] = cls._align_hydrofabric_values(
                initial_depth, network, 'initial_depth'
            )
            model.save_state()
        return model

    def load_hydrofabric(self, source, reach_ids=None):
        """Load geometry for a model whose topology matches a Hydrofabric."""
        network = load_hydrofabric(source, reach_ids=reach_ids)
        model_ids = tuple(str(value) for value in self.reach_ids)
        if network.reach_ids != model_ids:
            raise ValueError('Hydrofabric flowpath order does not match the '
                             'model; use MuskingumCunge.from_hydrofabric()')
        if not np.array_equal(network.endnodes, self.endnodes):
            raise ValueError('Hydrofabric topology does not match the model')
        self.set_geometry({
            'So': network.So,
            'dx': network.dx,
            'n': network.n,
            'Cs': network.Cs,
            'Bw': network.Bw,
            'Tw': network.Tw,
            'TwCC': network.TwCC,
            'nCC': network.nCC,
        })
        self.hydrofabric_network = network
        return self

    # ------------------------------------------------------------------
    # Geometry loading
    # ------------------------------------------------------------------
    def set_geometry(self, geometry):
        geometry = dict(geometry)
        # Finessing related to multiple variables with n
        if 'mann_n' not in geometry and 'n' in geometry:
            geometry['mann_n'] = geometry['n']
        n = self.n
        for field in _GEOM_FIELDS:
            if field == 'dx' and field not in geometry and self.dx is not None:
                continue
            if field not in geometry:
                raise ValueError(f'Geometry missing required field `{field}`')
            arr = np.asarray(geometry[field], dtype=np.float64)
            if arr.size != n:
                raise ValueError(f'Geometry field `{field}` has length '
                                 f'{arr.size}, expected {n}')
            setattr(self, field, arr.copy())
        # Zero bed slope makes the hydraulic solve break.
        self.So = np.maximum(self.So, MIN_SLOPE)
        self._validate_geometry()

    def load_routelink(self, routelink, id_field='link'):
        """
        Load and align hydraulic geometry from a RouteLink table.

        Parameters
        ----------
        routelink : pandas.DataFrame
            RouteLink table
        id_field : str
            Name of the id column in `routelink`.
        """
        rl = routelink.copy()
        rl[id_field] = rl[id_field].astype(str)
        rl = rl.set_index(id_field)
        reach_ids = [str(r) for r in self.reach_ids]
        missing = set(reach_ids) - set(rl.index)
        if missing:
            raise ValueError(f'{len(missing)} reach id(s) not found in '
                             f'RouteLink, e.g. {sorted(missing)[:5]}')
        rl = rl.loc[reach_ids]
        geometry = {}
        for rl_col, attr in ROUTELINK_MAP.items():
            if rl_col in rl.columns:
                geometry[attr] = rl[rl_col].values.astype(np.float64)
        self.set_geometry(geometry)
        return self

    def _validate_geometry(self):
        for f in _GEOM_FIELDS:
            if getattr(self, f) is None:
                raise ValueError(f'Geometry field `{f}` is not set. Call '
                                 f'set_geometry or load_routelink first.')

    # Linear Muskingum compatibility
    def compute_muskingum_coeffs(self, *args, **kwargs):
        """Maintain the base model's coefficient arrays for current K and X."""
        try:
            super().compute_muskingum_coeffs(*args, **kwargs)
        except Exception:
            self.alpha[:] = 0.0
            self.beta[:] = 0.0
            self.chi[:] = 0.0
            self.gamma[:] = 1.0

    # Tack on depth to existing state framework
    def save_state(self):
        super().save_state()
        if hasattr(self, 'depth'):
            self.saved_states['depth'] = self.depth.copy()
        if hasattr(self, 'K_t_prev'):
            self.saved_states['K'] = self.K.copy()
            self.saved_states['X'] = self.X.copy()
            self.saved_states['K_t_prev'] = self.K_t_prev.copy()
            self.saved_states['X_t_prev'] = self.X_t_prev.copy()
            self.saved_states['coefficient_dt'] = self._coefficient_dt
        if hasattr(self, 'K_history'):
            self.K_history[self.datetime] = self.K.copy()
            self.X_history[self.datetime] = self.X.copy()

    def load_state(self):
        super().load_state()
        if 'depth' in self.saved_states:
            self.depth = self.saved_states['depth'].copy()
        if 'K' in self.saved_states:
            self.K = self.saved_states['K'].copy()
            self.X = self.saved_states['X'].copy()
            self.K_t_prev = self.saved_states['K_t_prev'].copy()
            self.X_t_prev = self.saved_states['X_t_prev'].copy()
            self._coefficient_dt = self.saved_states['coefficient_dt']
            self.compute_muskingum_coeffs(K=self.K, X=self.X,
                                           dt=self._coefficient_dt)

    def state_transition_jacobian(self):
        """Return the frozen-coefficient Jacobian for the latest MC substep.

        With ``assume_short_ts=False``, the current alpha, beta, and chi arrays
        are C2, C1, and C3, respectively, and the frozen Jacobian has the same
        recursive topology as the original linear Muskingum transition.
        ``assume_short_ts=True`` instead has only diagonal and direct-upstream
        dependencies.

        K and X themselves depend on the hydraulic state. Holding their latest
        values fixed is the local linearization used by the EKF.
        """
        if self.assume_short_ts:
            F = np.zeros((self.n, self.n), dtype=np.float64)
            indices = np.arange(self.n)
            F[indices, indices] = self.chi
            for upstream, downstream in enumerate(self.endnodes):
                if upstream != downstream:
                    F[downstream, upstream] += (
                        self.alpha[downstream] + self.beta[downstream]
                    )
            self.A = F
            return F

        # Diagnostic/test path only. Production covariance propagation remains
        # matrix-free and does not construct this dense matrix.
        F = np.zeros((self.n, self.n), dtype=np.float64)
        remaining = self.indegree.copy()
        headwaters = self.startnodes[remaining == 0]
        for headwater in headwaters:
            reach = headwater
            while remaining[reach] == 0:
                downstream = self.endnodes[reach]
                F[reach, reach] = self.chi[reach]
                if reach != downstream:
                    F[downstream, reach] += self.beta[downstream]
                    F[downstream] += self.alpha[downstream] * F[reach]
                remaining[downstream] -= 1
                reach = downstream
        self.A = F
        return F.copy()

    # Routing
    def step_iter(self, p_t_next, timedelta=None):
        """
        Advance one timestep using Muskingum-Cunge.

        `p_t_next` is the per-reach lateral inflow right now
        """
        self._validate_geometry()

        if timedelta is None:
            timedelta = self.timedelta
            dt = self.dt
        else:
            dt = float(timedelta.seconds)

        endnodes = self.endnodes
        indegree = self.indegree
        sub_startnodes = self.startnodes[indegree == 0]

        o_t_prev = self.o_t_next
        depthp = self.depth

        for _, callback in self.callbacks.items():
            callback.__on_step_start__()

        ql = np.asarray(p_t_next, dtype=np.float64)

        # Nudging callbacks publish the previous downstream nudge so NWM
        # equation 4.7 can add it to both upstream-flow terms at that reach.
        previous_nudge = np.zeros(self.n, dtype=np.float64)
        for callback in self.callbacks.values():
            candidate = getattr(callback, 'routing_nudge', None)
            if candidate is None:
                continue
            candidate = np.asarray(candidate, dtype=np.float64)
            if candidate.shape != (self.n,):
                raise ValueError('Callback routing_nudge must have one value '
                                 'per reach')
            previous_nudge += candidate

        qdc, velc, depthc, qup_acc, quc_acc, Kc, Xc = _mc_ax_bu(
            sub_startnodes, endnodes, indegree, o_t_prev, ql, depthp,
            dt, self.So, self.dx, self.mann_n, self.Cs,
            self.Bw, self.Tw, self.TwCC, self.nCC,
            previous_nudge, self.assume_short_ts,
        )

        # Update states
        self.K_t_prev = self.K.copy()
        self.X_t_prev = self.X.copy()
        self.K = Kc
        self.X = Xc
        self._coefficient_dt = dt
        self.compute_muskingum_coeffs(K=Kc, X=Xc, dt=dt)
        self.i_t_prev = self.i_t_next
        self.i_t_next = quc_acc          # accumulated upstream inflow
        self.o_t_prev = o_t_prev
        self.o_t_next = qdc
        self.depth = depthc
        self.velocity = velc
        self.datetime += timedelta
        self.K_history[self.datetime] = Kc.copy()
        self.X_history[self.datetime] = Xc.copy()

        for _, callback in self.callbacks.items():
            callback.__on_step_end__()
        self.logger.debug(f'Stepped to time {self.datetime}')
