"""Kalman Filter and Extended Kalman Filter"""
import os
import copy
import datetime
import logging
import numpy as np
import pandas as pd
import scipy.linalg
from tx_fast_hydrology.nutils import interpolate_sample, interpolate_samples
from tx_fast_hydrology.nutils import (
    _ap_par, _aqat_par, _apply_gain, _short_ts_aqat_par,
)
from tx_fast_hydrology.callbacks import BaseCallback
from tx_fast_hydrology.nudging import (
    StreamflowNudging,
    TRouteStreamflowNudging,
    WRFHydroStreamflowNudging,
)

logger = logging.getLogger(__name__)

class KalmanFilter(BaseCallback):
    def __init__(self, model, measurements, Q_cov, R_cov, P_t_init):
        self.model = model
        self.measurements = measurements
        self.Q_cov = Q_cov
        self.R_cov = R_cov
        self.P_t_next = P_t_init
        self.P_t_prev = P_t_init
        self.reach_ids = model.reach_ids
        self.gage_reach_ids = measurements.columns
        self.datetime = copy.deepcopy(model.datetime)
        self.num_measurements = measurements.shape[1]

        assert isinstance(measurements.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert (measurements.index.tz == datetime.timezone.utc)
        assert np.isin(measurements.columns, self.reach_ids).all()

        reach_index_map = pd.Series(np.arange(len(self.reach_ids)), index=self.reach_ids)
        reach_indices = reach_index_map.loc[self.gage_reach_ids].values.astype(int)
        s = np.zeros(model.n, dtype=bool)
        s[reach_indices] = True
        self.s = s
        # NOTE: Sorting of input data done here
        permutations = np.argsort(reach_indices)
        is_sorted = (permutations == np.arange(self.num_measurements)).all()
        if not is_sorted:
            logger.warning('Measurement indices not sorted. Permuting columns...')
        reach_indices = reach_indices[permutations]
        self.reach_indices = reach_indices
        self.measurements = self.measurements.iloc[:, permutations]
        self.R_cov = self.R_cov[permutations, :][:, permutations]

        self.saved_states = {}
        self.save_state()

    def __on_simulation_start__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    def __on_step_end__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    @property
    def latest_measurement(self):
        return self.measurements.iloc[-1, :].values

    @property
    def latest_timestamp(self):
        return self.measurements.index[-1]

    def interpolate_input(self, datetime, method='linear'):
        datetime = float(datetime.value)
        datetimes = self.measurements.index.astype('int64').astype(float).values
        samples = self.measurements.values
        if method == 'linear':
            method_code = 1
        elif method == 'nearest':
            method_code = 0
        else:
            raise ValueError
        return interpolate_sample(datetime, datetimes, samples, method=method_code)

    def save_state(self):
        self.saved_states["datetime"] = self.datetime
        self.saved_states["P_t_next"] = self.P_t_next.copy()

    def load_state(self):
        self.datetime = self.saved_states["datetime"]
        self.P_t_next = self.saved_states["P_t_next"]

    def filter(self):
        # Implements KF after step is called
        P_t_prev = self.P_t_next
        i_t_next = self.model.i_t_next
        o_t_next = self.model.o_t_next
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        measurements = self.measurements
        s = self.s
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        datetime = self.model.datetime
        # Computed parameters
        Z = self.interpolate_input(datetime)
        #Z = measurements.loc[datetime].values
        dz = Z - o_t_next[s]
        # Compute prior covariance
        out = np.empty(P_t_prev.shape)
        P_t_next = _aqat_par(P_t_prev, out, sub_startnodes, endnodes,
                             alpha, beta, chi, indegree)
        P_t_next += Q_cov
        # Compute gain and posterior covariance
        K = P_t_next[:,s] @ np.linalg.inv(P_t_next[s][:,s] + R_cov)
        # K = np.linalg.solve(P_t_next[s][:,s] + R_cov, P_t_next[s, :]).T
        gain = K @ dz
        P_t_next = P_t_next - K @ P_t_next[s]
        # Apply gain
        i_t_gain, o_t_gain = _apply_gain(sub_startnodes, endnodes, gain, indegree)
        i_t_next += i_t_gain
        o_t_next += o_t_gain
        # Save posterior estimates
        self.model.i_t_next = i_t_next
        self.model.o_t_next = o_t_next        
        self.P_t_next = P_t_next
        self.P_t_prev = P_t_prev
        self.K = K
        self.dz = dz
        self.gain = gain
        # Update time
        self.datetime = datetime


class ExtendedKalmanFilter(KalmanFilter):
    """Frozen-coefficient EKF for Muskingum-Cunge models.

    The nonlinear routing step updates K and X before this callback runs.
    Those coefficients are held fixed for the covariance prediction. The
    topological mode reuses the original matrix-free ``_aqat_par`` traversal;
    the short-timestep mode uses its direct-upstream counterpart. Neither path
    constructs a dense transition matrix.
    """

    def __init__(self, model, measurements, Q_cov, R_cov, P_t_init):
        if not callable(getattr(model, 'state_transition_jacobian', None)):
            raise TypeError('ExtendedKalmanFilter requires a model with '
                            'state_transition_jacobian()')

        Q_cov = np.asarray(Q_cov, dtype=np.float64)
        R_cov = np.asarray(R_cov, dtype=np.float64)
        P_t_init = np.asarray(P_t_init, dtype=np.float64)
        expected_state_shape = (model.n, model.n)
        expected_measurement_shape = (
            measurements.shape[1], measurements.shape[1]
        )
        if P_t_init.shape != expected_state_shape:
            raise ValueError(f'P_t_init has shape {P_t_init.shape}; '
                             f'expected {expected_state_shape}')
        if Q_cov.shape != expected_state_shape:
            raise ValueError(f'Q_cov has shape {Q_cov.shape}; '
                             f'expected {expected_state_shape}')
        if R_cov.shape != expected_measurement_shape:
            raise ValueError(f'R_cov has shape {R_cov.shape}; '
                             f'expected {expected_measurement_shape}')

        super().__init__(model, measurements, Q_cov, R_cov, P_t_init.copy())
        self.F = None
        self.covariance_trace_history = {}

    def filter(self):
        P_t_prev = self.P_t_next
        i_t_next = self.model.i_t_next.copy()
        o_t_next = self.model.o_t_next.copy()
        datetime = self.model.datetime

        observations = self.interpolate_input(datetime)
        valid = np.isfinite(observations)
        observation_indices = self.reach_indices[valid]
        dz = observations[valid] - o_t_next[observation_indices]

        # Apply the selected frozen routing operator directly to covariance;
        # the dense Jacobian is reserved for diagnostics and small tests.
        out = np.empty(P_t_prev.shape, dtype=np.float64)
        if self.model.assume_short_ts:
            P_t_prior = _short_ts_aqat_par(
                P_t_prev, out, self.model.endnodes,
                self.model.alpha, self.model.beta, self.model.chi,
            )
        else:
            sub_startnodes = self.model.startnodes[
                self.model.indegree == 0
            ]
            P_t_prior = _aqat_par(
                P_t_prev, out, sub_startnodes, self.model.endnodes,
                self.model.alpha, self.model.beta, self.model.chi,
                self.model.indegree,
            )
        P_t_prior += self.Q_cov
        P_t_prior = 0.5 * (P_t_prior + P_t_prior.T)

        gain = np.zeros(self.model.n, dtype=np.float64)
        if observation_indices.size:
            # H is a row selector, so its products reduce to covariance slices.
            R_cov = self.R_cov[np.ix_(valid, valid)]
            innovation_cov = (
                P_t_prior[np.ix_(observation_indices, observation_indices)]
                + R_cov
            )
            K = np.linalg.solve(
                innovation_cov, P_t_prior[observation_indices, :]
            ).T
            gain = K @ dz
            P_t_next = (
                P_t_prior - K @ P_t_prior[observation_indices, :]
            )
            P_t_next = 0.5 * (P_t_next + P_t_next.T)
        else:
            K = np.empty((self.model.n, 0), dtype=np.float64)
            P_t_next = P_t_prior

        sub_startnodes = self.model.startnodes[(self.model.indegree == 0)]
        i_t_gain, o_t_gain = _apply_gain(
            sub_startnodes, self.model.endnodes, gain,
            self.model.indegree,
        )
        self.model.i_t_next = i_t_next + i_t_gain
        self.model.o_t_next = o_t_next + o_t_gain

        self.P_t_prior = P_t_prior
        self.P_t_next = P_t_next
        self.P_t_prev = P_t_prev
        self.observation_indices = observation_indices
        self.K = K
        self.dz = dz
        self.gain = gain
        self.datetime = datetime
        self.F = None
        self.covariance_trace_history[datetime] = float(np.trace(P_t_next))


class KalmanSmoother(KalmanFilter):
    def __init__(self, model, measurements, Q_cov, R_cov, P_t_init):
        super().__init__(model, measurements, Q_cov, R_cov, P_t_init)
        N = len(measurements) - 1
        self.N = N
        self.datetimes = [model.datetime]
        self.P_f = {}
        self.P_p = {}
        self.i_hat_f = {}
        self.o_hat_f = {}
        self.i_hat_f[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_f[model.datetime] = self.model.o_t_next.copy()
        self.i_hat_p = {}
        self.o_hat_p = {}
        self.i_hat_p[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_p[model.datetime] = self.model.o_t_next.copy()
        self.P_f[model.datetime] = self.P_t_next.copy()
        self.P_p[model.datetime] = self.P_t_next.copy()
        self.i_hat_s = None
        self.o_hat_s = None

    def __on_step_end__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    def __on_simulation_end__(self):
        return self.smooth()
    
    def filter(self):
        # Implements KF after step is called
        P_t_prev = self.P_t_next
        i_t_prior = self.model.i_t_next
        o_t_prior = self.model.o_t_next
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        measurements = self.measurements
        s = self.s
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        datetime = self.model.datetime
        # Computed parameters
        Z = measurements.loc[datetime].values
        dz = Z - o_t_prior[s]
        # Compute prior covariance
        out = np.empty(P_t_prev.shape)
        P_t_prior = _aqat_par(P_t_prev, out, sub_startnodes, endnodes,
                              alpha, beta, chi, indegree)
        P_t_prior += Q_cov
        # Compute gain and posterior covariance
        K = P_t_prior[:,s] @ np.linalg.inv(P_t_prior[s][:,s] + R_cov)
        gain = K @ dz
        P_t_next = P_t_prior - K @ P_t_prior[s]
        # Apply gain
        i_t_gain, o_t_gain = _apply_gain(sub_startnodes, endnodes, gain, indegree)
        i_t_next = i_t_prior + i_t_gain
        o_t_next = o_t_prior + o_t_gain
        # Save posterior estimates
        self.model.i_t_next = i_t_next
        self.model.o_t_next = o_t_next        
        self.P_t_prev = P_t_prev
        self.P_t_next = P_t_next
        # Save outputs
        self.i_hat_f[datetime] = i_t_next
        self.i_hat_p[datetime] = i_t_prior
        self.o_hat_f[datetime] = o_t_next
        self.o_hat_p[datetime] = o_t_prior
        self.P_p[datetime] = P_t_prior
        self.P_f[datetime] = P_t_next
        self.datetimes.append(datetime)
        # Update time
        self.datetime = datetime

    def smooth(self):
        P_p = self.P_p
        P_f = self.P_f
        datetimes = self.datetimes
        N = len(datetimes) - 1
        i_hat_f = self.i_hat_f
        o_hat_f = self.o_hat_f
        i_hat_p = self.i_hat_p
        o_hat_p = self.o_hat_p
        P_s = {}
        i_hat_s = {}
        o_hat_s = {}
        P_s[self.datetime] = P_f[self.datetime]
        i_hat_s[self.datetime] = i_hat_f[self.datetime]
        o_hat_s[self.datetime] = o_hat_f[self.datetime]
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        # Make this more efficient later
        A = self.model.A
        for k in reversed(range(N)):
            t = datetimes[k]
            tp1 = datetimes[k+1]
            P_f_t = P_f[t]
            out = np.empty(P_f_t.shape)
            A_Pf = _ap_par(P_f_t, out, sub_startnodes, endnodes, alpha, beta, chi, indegree)
            J = np.linalg.solve(P_p[tp1], A_Pf).T
            i_t_s = i_hat_f[t] + J @ (i_hat_s[tp1] - i_hat_p[tp1])
            o_t_s = o_hat_f[t] + J @ (o_hat_s[tp1] - o_hat_p[tp1])
            P = P_f[t] + J @ (P_s[tp1] - P_p[tp1])
            i_hat_s[t] = i_t_s
            o_hat_s[t] = o_t_s
            P_s[t] = P
        i_hat_s = pd.DataFrame.from_dict(i_hat_s, orient='index')
        i_hat_s.columns = self.model.reach_ids
        o_hat_s = pd.DataFrame.from_dict(o_hat_s, orient='index')
        o_hat_s.columns = self.model.reach_ids
        self.i_hat_s = i_hat_s
        self.o_hat_s = o_hat_s
        self.P_s = P_s


class KalmanSmootherIO(KalmanSmoother):
    def __init__(self, model, measurements, Q_cov, R_cov, P_t_init, temp_file):
        super(KalmanSmoother, self).__init__(model, measurements, Q_cov, R_cov, P_t_init)
        epoch = str(model.datetime.value)
        N = len(measurements) - 1
        self.N = N
        self.datetimes = [model.datetime]
        self.i_hat_f = {}
        self.o_hat_f = {}
        self.i_hat_p = {}
        self.o_hat_p = {}
        self.i_hat_f[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_f[model.datetime] = self.model.o_t_next.copy()
        self.i_hat_p[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_p[model.datetime] = self.model.o_t_next.copy()
        self.temp_file = temp_file
        P_f = pd.DataFrame(self.P_t_next)
        P_p = pd.DataFrame(self.P_t_next)
        P_f.to_hdf(f'{temp_file}', key=f'Pf__{epoch}', mode='a')
        P_p.to_hdf(f'{temp_file}', key=f'Pp__{epoch}', mode='a')
        self.i_hat_s = None
        self.o_hat_s = None

    def __on_simulation_end__(self):
        self.smooth()
        os.remove(self.temp_file)

    def filter(self):
        # Implements KF after step is called
        P_t_prev = self.P_t_next
        i_t_prior = self.model.i_t_next
        o_t_prior = self.model.o_t_next
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        measurements = self.measurements
        s = self.s
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        datetime = self.model.datetime
        epoch = str(self.model.datetime.value)
        # Computed parameters
        Z = measurements.loc[datetime].values
        dz = Z - o_t_prior[s]
        # Compute prior covariance
        out = np.empty(P_t_prev.shape)
        P_t_prior = _aqat_par(P_t_prev, out, sub_startnodes, endnodes,
                              alpha, beta, chi, indegree)
        P_t_prior += Q_cov
        # Compute gain and posterior covariance
        K = P_t_prior[:,s] @ np.linalg.inv(P_t_prior[s][:,s] + R_cov)
        gain = K @ dz
        P_t_next = P_t_prior - K @ P_t_prior[s]
        # Apply gain
        i_t_gain, o_t_gain = _apply_gain(sub_startnodes, endnodes, gain, indegree)
        i_t_next = i_t_prior + i_t_gain
        o_t_next = o_t_prior + o_t_gain
        # Save posterior estimates
        self.model.i_t_next = i_t_next
        self.model.o_t_next = o_t_next        
        self.P_t_prev = P_t_prev
        self.P_t_next = P_t_next
        # Save outputs
        self.i_hat_f[datetime] = i_t_next
        self.i_hat_p[datetime] = i_t_prior
        self.o_hat_f[datetime] = o_t_next
        self.o_hat_p[datetime] = o_t_prior
        pd.DataFrame(P_t_prior).to_hdf(f'{self.temp_file}', key=f'Pp__{epoch}', mode='a')
        pd.DataFrame(P_t_next).to_hdf(f'{self.temp_file}', key=f'Pf__{epoch}', mode='a')
        self.datetimes.append(datetime)
        # Update time
        self.datetime = datetime

    def smooth(self):
        temp_file = self.temp_file
        datetimes = self.datetimes
        N = len(datetimes) - 1
        i_hat_f = self.i_hat_f
        o_hat_f = self.o_hat_f
        i_hat_p = self.i_hat_p
        o_hat_p = self.o_hat_p
        i_hat_s = {}
        o_hat_s = {}
        P_s_tp1 = pd.read_hdf(f'{temp_file}', key=f'Pf__{self.datetime.value}').values
        i_hat_s[self.datetime] = i_hat_f[self.datetime]
        o_hat_s[self.datetime] = o_hat_f[self.datetime]
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        for k in reversed(range(N)):
            t = datetimes[k]
            tp1 = datetimes[k+1]
            P_f_t = pd.read_hdf(f'{temp_file}', key=f'Pf__{t.value}').values
            P_p_tp1 = pd.read_hdf(f'{temp_file}', key=f'Pp__{tp1.value}').values
            out = np.empty(P_f_t.shape)
            A_Pf = _ap_par(P_f_t, out, sub_startnodes, endnodes, alpha, beta, chi, indegree)
            J = np.linalg.solve(P_p_tp1, A_Pf).T
            i_t_s = i_hat_f[t] + J @ (i_hat_s[tp1] - i_hat_p[tp1])
            o_t_s = o_hat_f[t] + J @ (o_hat_s[tp1] - o_hat_p[tp1])
            P = P_f_t + J @ (P_s_tp1 - P_p_tp1)
            i_hat_s[t] = i_t_s
            o_hat_s[t] = o_t_s
            P_s_tp1 = P
        i_hat_s = pd.DataFrame.from_dict(i_hat_s, orient='index')
        i_hat_s.columns = self.model.reach_ids
        o_hat_s = pd.DataFrame.from_dict(o_hat_s, orient='index')
        o_hat_s.columns = self.model.reach_ids
        self.i_hat_s = i_hat_s
        self.o_hat_s = o_hat_s


class ReservoirNudging(BaseCallback):
    """Replace reservoir outlet flow with time-interpolated RFC forecasts."""

    def __init__(self, model, measurements):
        self.model = model
        self.measurements = measurements
        self.datetime = copy.deepcopy(model.datetime)

        assert isinstance(
            measurements.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        assert (measurements.index.tz == datetime.timezone.utc)
        assert (measurements.shape[1] == 1)

        self.saved_states = {}
        self.save_state()

    def __on_simulation_start__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    def __on_step_end__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    @property
    def latest_measurement(self):
        return self.measurements.iloc[-1, :].values

    @property
    def latest_timestamp(self):
        return self.measurements.index[-1]

    def interpolate_input(self, datetime, method='linear'):
        datetime = float(datetime.value)
        # Timestamp.value is always nanoseconds, even when a newer Pandas
        # version stores the DatetimeIndex at microsecond resolution.
        datetimes = self.measurements.index.as_unit('ns').asi8.astype(float)
        samples = self.measurements.values
        if method == 'linear':
            method_code = 1
        elif method == 'nearest':
            method_code = 0
        else:
            raise ValueError
        return interpolate_sample(
            datetime, datetimes, samples, method=method_code
        )

    def save_state(self):
        self.saved_states["datetime"] = self.datetime

    def load_state(self):
        self.datetime = self.saved_states["datetime"]

    def filter(self):
        datetime = self.model.datetime
        # Computed parameters
        outlet_index = self.model.outlet_index
        Z = self.interpolate_input(datetime).item()
        # Save posterior estimates
        self.model.o_t_next[outlet_index] = Z
        # Update time
        self.datetime = datetime
