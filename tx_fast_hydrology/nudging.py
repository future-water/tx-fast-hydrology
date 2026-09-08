"""t-route and WRF-Hydro streamflow nudging for MC routing."""

import numpy as np
import pandas as pd

from tx_fast_hydrology.callbacks import BaseCallback


class WRFHydroStreamflowNudging(BaseCallback):
    """Apply local NWM streamflow nudging at collocated channel reaches.

    This callback implements equations 4.1-4.4 and 4.7 of the WRF-Hydro
    v5.1.1 Technical Description. Observations are combined with squared
    inverse-time weights inside ``+/- tau`` and may be persisted into a
    forecast using flow-dependent coefficients from ``nudgingParams.nc``.

    The current nudge is added to modeled discharge after routing. It becomes
    part of the saved downstream-flow state, and ``routing_nudge`` exposes it
    to ``MuskingumCunge`` for the two equation 4.7 upstream-flow terms.

    Parameters
    ----------
    model : Muskingum or MuskingumCunge
        Bound linear or Muskingum-Cunge routing model.
    measurements : pandas.DataFrame
        Discharge observations in m3/s. Columns identify gages unless
        ``reach_ids`` is omitted, in which case they identify model reaches.
    reach_ids : sequence, optional
        Model reach collocated with each measurement column.
    quality : scalar or pandas.DataFrame, default 1.0
        Observation quality multiplier in [0, 1].
    G : scalar or sequence, default 1.0
        Per-gage nudging amplitude.
    tau_minutes : scalar or sequence, default 15
        Half-width of the direct-assimilation window.
    q_thresholds : array, optional
        Shape ``(gage, 12, threshold)``. Defaults to the NWM v2.1 value -100.
    exp_coefficients : array or scalar, default 120
        Persistence e-folding time in minutes, with shape
        ``(gage, 12, threshold + 1)`` when supplied as an array.
    temporal_persistence : bool, default True
        Persist the latest observation outside the direct window.
    assimilation_end : timestamp, optional
        Ignore observations after this time while retaining persistence.
    """

    def __init__(self, model, measurements, *, reach_ids=None, quality=1.0,
                 G=1.0, tau_minutes=15.0, q_thresholds=None,
                 exp_coefficients=120.0, temporal_persistence=True,
                 assimilation_end=None, minimum_flow=1.0e-10):
        self.model = model
        self.measurements = self._normalize_measurements(measurements)
        self.measurement_ids = self.measurements.columns.astype(str).tolist()
        self.measurements.columns = self.measurement_ids
        self.num_measurements = len(self.measurement_ids)

        if reach_ids is None:
            reach_ids = self.measurement_ids
        reach_ids = [str(value) for value in reach_ids]
        if len(reach_ids) != self.num_measurements:
            raise ValueError('reach_ids must have one entry per measurement '
                             'column')
        if len(set(reach_ids)) != len(reach_ids):
            raise ValueError('Only one stream gage may be collocated with a '
                             'reach')

        model_index = {str(value): index
                       for index, value in enumerate(model.reach_ids)}
        missing = [value for value in reach_ids if value not in model_index]
        if missing:
            raise ValueError(f'Gage reach IDs are absent from the model: '
                             f'{missing[:5]}')
        self.reach_ids = reach_ids
        self.reach_indices = np.asarray(
            [model_index[value] for value in reach_ids], dtype=np.int64,
        )

        self.quality = self._normalize_quality(quality)
        self.G = self._gage_vector(G, 'G')
        self.tau_minutes = self._gage_vector(
            tau_minutes, 'tau_minutes', positive=True,
        )
        self.q_thresholds = self._threshold_array(q_thresholds)
        self.exp_coefficients = self._coefficient_array(exp_coefficients)
        self.temporal_persistence = bool(temporal_persistence)
        self.minimum_flow = float(minimum_flow)
        if self.minimum_flow < 0:
            raise ValueError('minimum_flow cannot be negative')

        self.assimilation_end = None
        if assimilation_end is not None:
            self.set_assimilation_end(assimilation_end)

        self.previous_nudge = np.zeros(model.n, dtype=np.float64)
        self.current_nudge = self.previous_nudge.copy()
        self.latest_observation_time = [pd.NaT] * self.num_measurements
        self.latest_observation = np.full(
            self.num_measurements, np.nan, dtype=np.float64,
        )
        self.latest_quality = np.zeros(
            self.num_measurements, dtype=np.float64,
        )
        self.nudge_history = {}
        self._last_applied_time = None
        self.saved_states = {}
        self.save_state()

    @staticmethod
    def _normalize_measurements(measurements):
        if not isinstance(measurements, pd.DataFrame):
            raise TypeError('measurements must be a pandas DataFrame')
        if not isinstance(measurements.index, pd.DatetimeIndex):
            raise TypeError('measurements must use a DatetimeIndex')
        frame = measurements.copy()
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize('UTC')
        else:
            frame.index = frame.index.tz_convert('UTC')
        frame = frame.sort_index()
        if frame.index.has_duplicates:
            frame = frame.loc[~frame.index.duplicated(keep='last')]
        if frame.columns.duplicated().any():
            raise ValueError('measurement columns must be unique')
        return frame.astype(np.float64)

    def _normalize_quality(self, quality):
        if np.isscalar(quality):
            values = np.full(
                self.measurements.shape, float(quality), dtype=np.float64,
            )
            result = pd.DataFrame(
                values, index=self.measurements.index,
                columns=self.measurement_ids,
            )
        elif isinstance(quality, pd.DataFrame):
            result = quality.copy()
            if not isinstance(result.index, pd.DatetimeIndex):
                raise TypeError('quality must use a DatetimeIndex')
            if result.index.tz is None:
                result.index = result.index.tz_localize('UTC')
            else:
                result.index = result.index.tz_convert('UTC')
            result.columns = result.columns.astype(str)
            result = result.reindex(
                index=self.measurements.index, columns=self.measurement_ids,
            ).astype(np.float64)
        else:
            raise TypeError('quality must be a scalar or pandas DataFrame')
        finite = result.to_numpy()[np.isfinite(result.to_numpy())]
        if finite.size and ((finite < 0).any() or (finite > 1).any()):
            raise ValueError('quality values must lie in [0, 1]')
        return result

    def _gage_vector(self, values, name, positive=False):
        if np.isscalar(values):
            result = np.full(
                self.num_measurements, float(values), dtype=np.float64,
            )
        else:
            result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.num_measurements,):
            raise ValueError(f'{name} must have one value per gage')
        if not np.isfinite(result).all():
            raise ValueError(f'{name} contains non-finite values')
        if positive and (result <= 0).any():
            raise ValueError(f'{name} values must be positive')
        return result.copy()

    def _threshold_array(self, values):
        if values is None:
            return np.full(
                (self.num_measurements, 12, 1), -100.0,
                dtype=np.float64,
            )
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 1 and result.size == self.num_measurements:
            result = np.broadcast_to(
                result[:, None, None],
                (self.num_measurements, 12, 1),
            )
        elif result.ndim == 2 and result.shape[0] == 12:
            result = np.broadcast_to(
                result[None, :, :],
                (self.num_measurements,) + result.shape,
            )
        if result.ndim != 3 or result.shape[:2] != (
                self.num_measurements, 12):
            raise ValueError('q_thresholds must have shape '
                             '(gage, 12, threshold)')
        if not np.isfinite(result).all():
            raise ValueError('q_thresholds contains non-finite values')
        return result.copy()

    def _coefficient_array(self, values):
        categories = self.q_thresholds.shape[2] + 1
        if np.isscalar(values):
            result = np.full(
                (self.num_measurements, 12, categories), float(values),
                dtype=np.float64,
            )
        else:
            result = np.asarray(values, dtype=np.float64)
            if result.ndim == 2 and result.shape == (12, categories):
                result = np.broadcast_to(
                    result[None, :, :],
                    (self.num_measurements, 12, categories),
                )
        expected = (self.num_measurements, 12, categories)
        if result.shape != expected:
            raise ValueError(f'exp_coefficients must have shape {expected}')
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError('exp_coefficients must be finite and positive')
        return result.copy()

    @classmethod
    def from_nwm_parameters(cls, model, measurements, parameter_file, *,
                            reach_ids=None, station_ids=None, **kwargs):
        """Construct from an operational ``nudgingParams.nc`` file."""
        import xarray as xr

        normalized = cls._normalize_measurements(measurements)
        measurement_ids = normalized.columns.astype(str).tolist()
        if station_ids is None:
            station_ids = measurement_ids
        station_ids = [str(value).strip() for value in station_ids]
        if len(station_ids) != len(measurement_ids):
            raise ValueError('station_ids must have one entry per measurement '
                             'column')

        with xr.open_dataset(parameter_file, decode_times=False) as dataset:
            available = [
                value.decode('ascii', errors='ignore').strip()
                if isinstance(value, (bytes, np.bytes_)) else str(value).strip()
                for value in dataset['stationId'].values
            ]
            lookup = {value: index for index, value in enumerate(available)}
            missing = [value for value in station_ids if value not in lookup]
            if missing:
                raise ValueError(f'Stations are absent from nudgingParams: '
                                 f'{missing[:5]}')
            indices = np.asarray([lookup[value] for value in station_ids])
            G = np.asarray(dataset['G'].values[indices], dtype=np.float64)
            tau = np.asarray(dataset['tau'].values[indices], dtype=np.float64)
            thresholds = np.asarray(
                dataset['qThresh'].values[indices], dtype=np.float64,
            )
            coefficients = np.asarray(
                dataset['expCoeff'].values[indices], dtype=np.float64,
            )

        return cls(
            model, normalized, reach_ids=reach_ids, G=G,
            tau_minutes=tau, q_thresholds=thresholds,
            exp_coefficients=coefficients, **kwargs,
        )

    @property
    def routing_nudge(self):
        """Return the previous nudge for the next MC routing substep."""
        return self.previous_nudge

    def set_assimilation_end(self, timestamp):
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        self.assimilation_end = timestamp
        return self

    def clear_assimilation_end(self):
        self.assimilation_end = None
        return self

    def __on_simulation_start__(self):
        self.update()

    def __on_step_end__(self):
        self.update()

    def __on_save_state__(self):
        self.save_state()

    def __on_load_state__(self):
        self.load_state()

    def _eligible_observation_mask(self, timestamp, gage_index):
        times = self.measurements.index
        cutoff = self.assimilation_end
        eligible = np.ones(len(times), dtype=bool)
        if cutoff is not None:
            eligible &= times <= cutoff
        delta_minutes = np.abs(
            (times - timestamp).total_seconds().to_numpy(dtype=np.float64)
        ) / 60.0
        eligible &= delta_minutes <= self.tau_minutes[gage_index]
        observations = self.measurements.iloc[:, gage_index].to_numpy()
        quality = self.quality.iloc[:, gage_index].to_numpy()
        eligible &= np.isfinite(observations) & np.isfinite(quality)
        return eligible, delta_minutes, observations, quality

    def _remember_latest_observations(self, timestamp):
        cutoff = timestamp
        if self.assimilation_end is not None:
            cutoff = min(cutoff, self.assimilation_end)
        times = self.measurements.index
        for gage_index in range(self.num_measurements):
            observations = self.measurements.iloc[:, gage_index].to_numpy()
            quality = self.quality.iloc[:, gage_index].to_numpy()
            valid = ((times <= cutoff) & np.isfinite(observations)
                     & np.isfinite(quality) & (quality > 0))
            positions = np.flatnonzero(valid)
            if not positions.size:
                continue
            position = int(positions[-1])
            candidate_time = times[position]
            current_time = self.latest_observation_time[gage_index]
            if pd.isna(current_time) or candidate_time > current_time:
                self.latest_observation_time[gage_index] = candidate_time
                self.latest_observation[gage_index] = observations[position]
                self.latest_quality[gage_index] = quality[position]

    def _persistence_nudge(self, gage_index, timestamp, model_discharge):
        observation_time = self.latest_observation_time[gage_index]
        if pd.isna(observation_time):
            return 0.0
        age_minutes = (timestamp - observation_time).total_seconds() / 60.0
        if age_minutes < 0:
            return 0.0
        month_index = timestamp.month - 1
        thresholds = self.q_thresholds[gage_index, month_index]
        category = int(np.count_nonzero(model_discharge > thresholds))
        coefficient = self.exp_coefficients[
            gage_index, month_index, category
        ]
        innovation = self.latest_observation[gage_index] - model_discharge
        return (self.latest_quality[gage_index] * innovation
                * np.exp(-age_minutes / coefficient))

    def _calculate_nudge(self, timestamp, discharge):
        result = np.zeros(self.model.n, dtype=np.float64)
        for gage_index, reach_index in enumerate(self.reach_indices):
            eligible, delta, observations, quality = (
                self._eligible_observation_mask(timestamp, gage_index)
            )
            modeled = float(discharge[reach_index])
            if eligible.any():
                tau = self.tau_minutes[gage_index]
                weights = 10.0 ** (-delta[eligible] / (tau / 10.0))
                weights_squared = weights * weights
                innovations = observations[eligible] - modeled
                nudge = (self.G[gage_index]
                         * np.sum(quality[eligible] * weights_squared
                                  * innovations)
                         / np.sum(weights_squared))
            elif self.temporal_persistence:
                nudge = self._persistence_nudge(
                    gage_index, timestamp, modeled,
                )
            else:
                nudge = 0.0
            if modeled + nudge < self.minimum_flow:
                nudge = self.minimum_flow - modeled
            result[reach_index] = nudge
        return result

    def update(self):
        """Calculate and apply the nudge at the model's current time."""
        timestamp = self.model.datetime
        if self._last_applied_time == timestamp:
            return self.current_nudge.copy()
        self._remember_latest_observations(timestamp)
        discharge = self.model.o_t_next.copy()
        nudge = self._calculate_nudge(timestamp, discharge)
        self.model.o_t_next = discharge + nudge
        self.previous_nudge = nudge.copy()
        self.current_nudge = nudge.copy()
        self.nudge_history[timestamp] = nudge.copy()
        self._last_applied_time = timestamp
        return nudge.copy()

    def save_state(self):
        self.saved_states = {
            'previous_nudge': self.previous_nudge.copy(),
            'current_nudge': self.current_nudge.copy(),
            'latest_observation_time': list(self.latest_observation_time),
            'latest_observation': self.latest_observation.copy(),
            'latest_quality': self.latest_quality.copy(),
            'last_applied_time': self._last_applied_time,
            'assimilation_end': self.assimilation_end,
        }

    def load_state(self):
        if not self.saved_states:
            return
        self.previous_nudge = self.saved_states['previous_nudge'].copy()
        self.current_nudge = self.saved_states['current_nudge'].copy()
        self.latest_observation_time = list(
            self.saved_states['latest_observation_time']
        )
        self.latest_observation = (
            self.saved_states['latest_observation'].copy()
        )
        self.latest_quality = self.saved_states['latest_quality'].copy()
        self._last_applied_time = self.saved_states['last_applied_time']
        self.assimilation_end = self.saved_states['assimilation_end']


class TRouteStreamflowNudging(BaseCallback):
    """Apply t-route's ``simple_da`` streamflow nudging algorithm.

    At an observation time, the modeled discharge at the reach is
    replaced by the observation. If the current observation is missing or the
    simulation has passed ``assimilation_end``, the most recent observation is
    blended with the current model value using

    ``exp(-abs(minutes_since_observation) / decay_coefficient)``.

    This follows ``simple_da.pyx`` and the call site in ``mc_reach.pyx``. The
    corrected discharge is retained as routing state, so it propagates through
    subsequent MC steps without WRF-Hydro's additional upstream-nudge term.

    Parameters
    ----------
    model : Muskingum or MuskingumCunge
        Bound linear or Muskingum-Cunge routing model.
    measurements : pandas.DataFrame
        Discharge observations in m3/s. As in t-route preprocessing, values
        are linearly interpolated to one-minute resolution so routing times
        between observation time slices receive an interpolated observation.
        Columns identify gages unless ``reach_ids`` is supplied.
    reach_ids : sequence, optional
        Model reach collocated with each measurement column.
    decay_coefficient : scalar or sequence, default 120
        Exponential e-folding time in minutes. t-route normally supplies one
        configuration value; a per-gage sequence is also accepted here.
    temporal_persistence : bool, default True
        Decay the last valid observation when no current observation exists.
    assimilation_end : timestamp, optional
        Ignore observations after this time while retaining persistence.
    quality : scalar or pandas.DataFrame, optional
        Observation quality in [0, 1]. Values below ``quality_threshold`` are
        discarded before interpolation; quality never scales the accepted
        discharge or nudge. Omit when measurements are already quality
        controlled.
    quality_threshold : float, default 1
        Minimum accepted quality, matching t-route's ``qc_threshold`` default.
    interpolation_limit_minutes : int or None, default 59
        Maximum consecutive one-minute values filled in either direction,
        matching t-route's observation preprocessing. Use ``None`` to disable
        interpolation and require exact observation timestamps.
    last_observation_values, last_observation_times : sequences, optional
        Restart state for each gage. Supply both or neither.
    """

    def __init__(self, model, measurements, *, reach_ids=None,
                 decay_coefficient=120.0, temporal_persistence=True,
                 assimilation_end=None, last_observation_values=None,
                 last_observation_times=None,
                 interpolation_limit_minutes=59, quality=None,
                 quality_threshold=1.0):
        self.model = model
        self.source_measurements = (
            WRFHydroStreamflowNudging._normalize_measurements(
                measurements
            )
        )
        self.measurement_ids = (
            self.source_measurements.columns.astype(str).tolist()
        )
        self.source_measurements.columns = self.measurement_ids
        self.quality_threshold = float(quality_threshold)
        if (not np.isfinite(self.quality_threshold)
                or not 0.0 <= self.quality_threshold <= 1.0):
            raise ValueError('quality_threshold must lie in [0, 1]')
        self.quality = self._normalize_quality(quality)
        if self.quality is not None:
            self.source_measurements = self.source_measurements.mask(
                self.quality.isna()
                | (self.quality < self.quality_threshold)
            )
        # t-route's timeslice QC also rejects zero and negative discharge.
        self.source_measurements = self.source_measurements.mask(
            self.source_measurements <= 0.0
        )
        self.measurements = self._interpolate_measurements(
            self.source_measurements, interpolation_limit_minutes,
        )
        self.num_measurements = len(self.measurement_ids)

        if reach_ids is None:
            reach_ids = self.measurement_ids
        reach_ids = [str(value) for value in reach_ids]
        if len(reach_ids) != self.num_measurements:
            raise ValueError('reach_ids must have one entry per measurement '
                             'column')
        if len(set(reach_ids)) != len(reach_ids):
            raise ValueError('Only one stream gage may be collocated with a '
                             'reach')
        model_index = {str(value): index
                       for index, value in enumerate(model.reach_ids)}
        missing = [value for value in reach_ids if value not in model_index]
        if missing:
            raise ValueError(f'Gage reach IDs are absent from the model: '
                             f'{missing[:5]}')
        self.reach_ids = reach_ids
        self.reach_indices = np.asarray(
            [model_index[value] for value in reach_ids], dtype=np.int64,
        )

        self.decay_coefficient = self._gage_vector(
            decay_coefficient, 'decay_coefficient', positive=True,
        )
        self.temporal_persistence = bool(temporal_persistence)
        self.assimilation_end = None
        if assimilation_end is not None:
            self.set_assimilation_end(assimilation_end)

        supplied_values = last_observation_values is not None
        supplied_times = last_observation_times is not None
        if supplied_values != supplied_times:
            raise ValueError('last_observation_values and '
                             'last_observation_times must be supplied together')
        if supplied_values:
            values = np.asarray(last_observation_values, dtype=np.float64)
            if values.shape != (self.num_measurements,):
                raise ValueError('last_observation_values must have one value '
                                 'per gage')
            times = list(last_observation_times)
            if len(times) != self.num_measurements:
                raise ValueError('last_observation_times must have one value '
                                 'per gage')
            self.latest_observation = values.copy()
            self.latest_observation_time = [
                pd.NaT if pd.isna(value) else self._as_utc_timestamp(value)
                for value in times
            ]
            mismatched = [
                np.isfinite(value) != (not pd.isna(timestamp))
                for value, timestamp in zip(
                    self.latest_observation, self.latest_observation_time,
                )
            ]
            if any(mismatched):
                raise ValueError('Each restart observation requires a matching '
                                 'time, and vice versa')
        else:
            self.latest_observation = np.full(
                self.num_measurements, np.nan, dtype=np.float64,
            )
            self.latest_observation_time = [pd.NaT] * self.num_measurements

        self.previous_nudge = np.zeros(model.n, dtype=np.float64)
        self.current_nudge = self.previous_nudge.copy()
        self.nudge_history = {}
        self._last_applied_time = None
        self.saved_states = {}
        self.save_state()

    @staticmethod
    def _interpolate_measurements(measurements, limit_minutes):
        if limit_minutes is None or measurements.empty:
            return measurements.copy()
        if isinstance(limit_minutes, bool):
            raise TypeError('interpolation_limit_minutes must be an integer '
                            'or None')
        limit_minutes = int(limit_minutes)
        if limit_minutes <= 0:
            raise ValueError('interpolation_limit_minutes must be positive')
        # This is the same sequence used by t-route's _interpolate_one:
        # resample to minutes, interpolate with a bounded gap, then the routing
        # layer selects values at its own timestep.
        return (
            measurements.resample('1min').asfreq().interpolate(
                limit=limit_minutes, limit_direction='both',
            )
        )

    def _normalize_quality(self, quality):
        if quality is None:
            return None
        if np.isscalar(quality):
            values = np.full(
                self.source_measurements.shape, float(quality),
                dtype=np.float64,
            )
            result = pd.DataFrame(
                values, index=self.source_measurements.index,
                columns=self.measurement_ids,
            )
        elif isinstance(quality, pd.DataFrame):
            result = quality.copy()
            if not isinstance(result.index, pd.DatetimeIndex):
                raise TypeError('quality must use a DatetimeIndex')
            if result.index.tz is None:
                result.index = result.index.tz_localize('UTC')
            else:
                result.index = result.index.tz_convert('UTC')
            result.columns = result.columns.astype(str)
            result = result.reindex(
                index=self.source_measurements.index,
                columns=self.measurement_ids,
            ).astype(np.float64)
        else:
            raise TypeError('quality must be a scalar, pandas DataFrame, or '
                            'None')
        finite = result.to_numpy()[np.isfinite(result.to_numpy())]
        if finite.size and ((finite < 0).any() or (finite > 1).any()):
            raise ValueError('quality values must lie in [0, 1]')
        return result

    @staticmethod
    def _as_utc_timestamp(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tz is None:
            return timestamp.tz_localize('UTC')
        return timestamp.tz_convert('UTC')

    def _gage_vector(self, values, name, positive=False):
        if np.isscalar(values):
            result = np.full(
                self.num_measurements, float(values), dtype=np.float64,
            )
        else:
            result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.num_measurements,):
            raise ValueError(f'{name} must have one value per gage')
        if not np.isfinite(result).all():
            raise ValueError(f'{name} contains non-finite values')
        if positive and (result <= 0).any():
            raise ValueError(f'{name} values must be positive')
        return result.copy()

    def set_assimilation_end(self, timestamp):
        self.assimilation_end = self._as_utc_timestamp(timestamp)
        return self

    def clear_assimilation_end(self):
        self.assimilation_end = None
        return self

    def __on_simulation_start__(self):
        self.update()

    def __on_step_end__(self):
        self.update()

    def __on_save_state__(self):
        self.save_state()

    def __on_load_state__(self):
        self.load_state()

    def _remember_latest_observations(self, timestamp):
        """Advance t-route's per-gage last-observation state."""
        cutoff = timestamp
        if self.assimilation_end is not None:
            cutoff = min(cutoff, self.assimilation_end)
        times = self.measurements.index
        for gage_index in range(self.num_measurements):
            observations = self.measurements.iloc[:, gage_index].to_numpy()
            valid = (times <= cutoff) & np.isfinite(observations)
            positions = np.flatnonzero(valid)
            if not positions.size:
                continue
            position = int(positions[-1])
            candidate_time = times[position]
            current_time = self.latest_observation_time[gage_index]
            if pd.isna(current_time) or candidate_time > current_time:
                self.latest_observation_time[gage_index] = candidate_time
                self.latest_observation[gage_index] = observations[position]

    def _current_observation(self, timestamp, gage_index):
        if (self.assimilation_end is not None
                and timestamp > self.assimilation_end):
            return np.nan
        # The preprocessing stage has already inserted interpolated timestamps,
        # so an exact lookup here mirrors the array indexing in t-route.
        position = self.measurements.index.get_indexer([timestamp])[0]
        if position < 0:
            return np.nan
        return float(self.measurements.iloc[position, gage_index])

    def _calculate_nudge(self, timestamp, discharge):
        result = np.zeros(self.model.n, dtype=np.float64)
        for gage_index, reach_index in enumerate(self.reach_indices):
            modeled = float(discharge[reach_index])
            observation = self._current_observation(timestamp, gage_index)
            if np.isfinite(observation):
                # t-route performs direct insertion at valid observation times.
                nudge = observation - modeled
            elif (self.temporal_persistence
                  and np.isfinite(self.latest_observation[gage_index])):
                observation_time = self.latest_observation_time[gage_index]
                age_minutes = abs(
                    (timestamp - observation_time).total_seconds() / 60.0
                )
                weight = np.exp(
                    -age_minutes / self.decay_coefficient[gage_index]
                )
                nudge = (
                    self.latest_observation[gage_index] - modeled
                ) * weight
            else:
                nudge = 0.0
            result[reach_index] = nudge
        return result

    def update(self):
        """Calculate and apply the t-route nudge at the current model time."""
        timestamp = self._as_utc_timestamp(self.model.datetime)
        if self._last_applied_time == timestamp:
            return self.current_nudge.copy()
        self._remember_latest_observations(timestamp)
        discharge = self.model.o_t_next.copy()
        nudge = self._calculate_nudge(timestamp, discharge)
        self.model.o_t_next = discharge + nudge
        self.previous_nudge = nudge.copy()
        self.current_nudge = nudge.copy()
        self.nudge_history[timestamp] = nudge.copy()
        self._last_applied_time = timestamp
        return nudge.copy()

    def save_state(self):
        self.saved_states = {
            'previous_nudge': self.previous_nudge.copy(),
            'current_nudge': self.current_nudge.copy(),
            'latest_observation_time': list(self.latest_observation_time),
            'latest_observation': self.latest_observation.copy(),
            'last_applied_time': self._last_applied_time,
            'assimilation_end': self.assimilation_end,
        }

    def load_state(self):
        if not self.saved_states:
            return
        self.previous_nudge = self.saved_states['previous_nudge'].copy()
        self.current_nudge = self.saved_states['current_nudge'].copy()
        self.latest_observation_time = list(
            self.saved_states['latest_observation_time']
        )
        self.latest_observation = (
            self.saved_states['latest_observation'].copy()
        )
        self._last_applied_time = self.saved_states['last_applied_time']
        self.assimilation_end = self.saved_states['assimilation_end']


class StreamflowNudging(TRouteStreamflowNudging):
    """Default streamflow nudging callback, using t-route ``simple_da``."""
