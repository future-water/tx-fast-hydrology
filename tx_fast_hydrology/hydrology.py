import sys
import math
import uuid
import numpy as np
import pandas as pd
from numba import njit
from scipy.integrate import odeint
import copy
from heapq import heappop, heappush

DEFAULT_START_TIME = pd.to_datetime(0., utc=True)
DEFAULT_TIMEDELTA = pd.to_timedelta(3600, unit='s')

# Adapted from cfe_py code at: https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py

class CFEModel():
    def __init__(self, data):
        self.load_model(data)
        self.soil_layer = SoilLayer(self, data) 
        self.groundwater_layer = GroundwaterLayer(self, data)

    @property
    def dt(self):
        return self.timedelta.seconds

    def step(self, p_t, pet_t, dt=None, num_iter=40, eps=1e-5):
        self.save_state()
        soil_layer = self.soil_layer
        groundwater_layer = self.groundwater_layer
        S_t_prev = self.soil_layer.S_t.copy()
        S_gw_t_prev = self.groundwater_layer.S_gw_t.copy()
        for _ in range(num_iter):
            # Rainfall and ET 
            soil_layer.calculate_evaporation_from_rainfall(p_t, pet_t)
            soil_layer.calculate_evaporation_from_soil(pet_t)
            
            # Infiltration partitioning
            soil_layer.calculate_infiltration_excess_runoff(p_t)
            soil_layer.calculate_lateral_flow_in_soil()
            soil_layer.calculate_percolation_from_soil()

            # Soil moisture reservoir
            soil_layer.calculate_soil_storage__trapezoidal(dt)

            # Groundwater model
            groundwater_layer.calculate_saturation_excess_overland_flow_from_gw()
            groundwater_layer.compute_groundwater_flux__exponential()  
            groundwater_layer.calculate_groundwater_storage__trapezoidal(dt)

            # Continue iterating until convergence
            soil_rel_err = soil_layer.S_t - S_t_prev
            gw_rel_err = groundwater_layer.S_gw_t - S_gw_t_prev
            max_rel_err = max(np.abs(soil_rel_err).max(), np.abs(gw_rel_err).max())
            if max_rel_err > eps:
                S_t_prev = soil_layer.S_t.copy()
                S_gw_t_prev = groundwater_layer.S_gw_t.copy()
            else:
                break
        self.soil_layer.calculate_surface_runoff__giuh()
        self.datetime = self.datetime + self.timedelta

    def save_state(self):
        self.soil_layer.save_state()
        self.groundwater_layer.save_state()

    def load_state(self):
        self.soil_layer.load_state()
        self.groundwater_layer.load_state()

    def load_model(self, obj, load_optional=True):
        required_fields = {'name', 'datetime', 'timedelta', 'watershed_ids'}
        optional_fields = set()
        defaults = {}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
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


class SoilLayer():
    def __init__(self, parent, data):
        self.parent = parent
        self.load_model(data)

        # Initialize simulation constants
        atm_press_Pa = 101325.
        unit_weight_water_N_per_m3 = 9810.

        # Local values to be used in setting up soil reservoir
        # TODO: Arbitrary initialization
        trigger_z_m = 0.5
        field_capacity_atm_press_fraction = self.alpha_fc

        # Soil reservoir configuration
        # Soil outflux calculation, Equation 3 in Fred Ogden's document
        H_water_table_m = (field_capacity_atm_press_fraction * atm_press_Pa 
                           / unit_weight_water_N_per_m3)

        soil_water_content_at_field_capacity = self.smcmax * np.power(
            H_water_table_m / self.satpsi, (1. / self.bb)
        )

        Omega = H_water_table_m - trigger_z_m
        # Upper & lower limit of the integral in Equation 4 in Fred Ogden's document
        lower_lim = np.power(Omega, (1. - 1. / self.bb)) / (1. - 1. / self.bb)
        upper_lim = np.power(Omega + self.D, (1. - 1. / self.bb)) / (1. - 1. / self.bb)
        # Integral & power term in Equation 4 & 5 in Fred Ogden's document
        storage_thresh_pow_term = np.power(1. / self.satpsi, (-1. / self.bb))
        lim_diff = upper_lim - lower_lim
        field_capacity_storage_threshold_m = (
            self.smcmax * storage_thresh_pow_term * lim_diff
        )

        self.S_thresh = field_capacity_storage_threshold_m
        self.S_wilt = self.smcwlt * self.D
        self.S_max = self.smcmax * self.D
        # TODO: Arbitrary initialization
        self.S_t = self.S_max * 2 / 3

        self.I_t = np.zeros(self.S_t.size, dtype=np.float64)
        self.et_soil_t = np.zeros(self.S_t.size, dtype=np.float64)
        self.Q_lf_t = np.zeros(self.S_t.size, dtype=np.float64)
        self.Q_perc_t = np.zeros(self.S_t.size, dtype=np.float64)
        self.Q_surf_t = np.zeros(self.S_t.size, dtype=np.float64)

        # Schaake partitioning
        self.refkdt = 3.0
        self.satdk_ref = 2e-6
        self.schaake_constant = self.refkdt * self.satdk / self.satdk_ref

        # TODO: Check this
        self.K_perc = self.satdk * self.slop

        # Runoff queues
        self.runoff_queues = [[] for _ in range(self.S_t.size)]
        self.Q_overflow_t = np.zeros(self.S_t.size, dtype=np.float64)

        # Nash cascade
        self.K_nash = 0.03  # Default value, but should be set in configuration file
        self.nash_queues = [[] for _ in range(self.S_t.size)]

        self.saved_states = {
            'datetime' : copy.copy(self.datetime),
            'S_t' : self.S_t.copy(),
            'I_t' : self.I_t.copy(),
            'et_soil_t' : self.et_soil_t.copy(),
            'Q_lf_t' : self.Q_lf_t.copy(),
            'Q_perc_t' : self.Q_perc_t.copy()
        }

    @property
    def datetime(self):
        return self.parent.datetime

    @property
    def timedelta(self):
        return self.parent.timedelta

    @property
    def dt(self):
        return self.parent.timedelta.seconds

    def save_state(self):
        self.saved_states['datetime'] = copy.copy(self.datetime)
        self.saved_states['S_t'] = self.S_t.copy()
        self.saved_states['I_t'] = self.I_t.copy()
        self.saved_states['et_soil_t'] = self.et_soil_t.copy()
        self.saved_states['Q_lf_t'] = self.Q_lf_t.copy()
        self.saved_states['Q_perc_t'] = self.Q_perc_t.copy()

    def load_state(self):
        self.datetime = self.saved_states['datetime']
        self.S_t = self.saved_states['S_t']

    def calculate_evaporation_from_rainfall(self, p_t, pet_t):
        et_rain_t = np.maximum(p_t, pet_t)
        self.et_rain_t = et_rain_t

    def calculate_evaporation_from_soil(self, pet_t):
        S_t = self.S_t
        S_thresh = self.S_thresh
        S_wilt = self.S_wilt
        et_soil_t = compute_et_from_soil(S_t, S_thresh, S_wilt, pet_t)
        self.et_soil_t = et_soil_t

    def calculate_lateral_flow_in_soil(self):
        S_t = self.S_t
        S_thresh = self.S_thresh
        S_max = self.S_max
        K_lf = self.K_lf
        Q_lf_t = compute_lateral_flow_in_soil(S_t, S_thresh, S_max, K_lf)
        self.Q_lf_t = Q_lf_t

    def calculate_percolation_from_soil(self):
        S_t = self.S_t
        S_thresh = self.S_thresh
        S_max = self.S_max
        K_perc = self.K_perc
        Q_perc_t = compute_percolation_from_soil(S_t, S_thresh, S_max, K_perc)
        self.Q_perc_t = Q_perc_t

    def calculate_infiltration_excess_runoff(self, p_t):
        S_t = self.S_t
        S_max = self.S_max
        schaake_constant = self.schaake_constant
        I_t = compute_infiltration_rate__schaake(S_t, p_t, S_max, schaake_constant)
        Q_surf_t = p_t - I_t
        self.I_t = I_t
        self.Q_surf_t = Q_surf_t

    def calculate_soil_storage__explicit(self, dt):
        if dt is None:
            dt = self.dt
        S_t_prev = self.saved_states['S_t']
        I_t = self.I_t
        et_soil_t = self.et_soil_t
        Q_lf_t = self.Q_lf_t
        Q_perc_t = self.Q_perc_t
        S_t_next = S_t_prev + dt * (I_t - et_soil_t - Q_lf_t - Q_perc_t)
        self.S_t = S_t_next

    def calculate_soil_storage__trapezoidal(self, dt):
        if dt is None:
            dt = self.dt
        S_t_prev = self.saved_states['S_t']
        I_t_prev = self.saved_states['I_t']
        et_soil_t_prev = self.saved_states['et_soil_t']
        Q_lf_t_prev = self.saved_states['Q_lf_t']
        Q_perc_t_prev = self.saved_states['Q_perc_t']
        I_t = self.I_t
        et_soil_t = self.et_soil_t
        Q_lf_t = self.Q_lf_t
        Q_perc_t = self.Q_perc_t
        f_prev = (I_t_prev - et_soil_t_prev - Q_lf_t_prev - Q_perc_t_prev)
        f_next = (I_t - et_soil_t - Q_lf_t - Q_perc_t)
        S_t_next = S_t_prev + dt / 2 * (f_prev + f_next)
        self.S_t = S_t_next

    def calculate_surface_runoff__giuh(self):
        dt = self.dt
        runoff_queues = self.runoff_queues
        yield_time = self.datetime + self.timedelta
        Q_surf_t = self.Q_surf_t
        giuh_timedeltas = self.giuh_timedeltas
        giuh_values = self.giuh_values
        Q_overflow_t = self.Q_overflow_t
        for i, queue in enumerate(runoff_queues):
            result = 0.
            times = giuh_timedeltas[i] + yield_time
            values = giuh_values[i] * Q_surf_t[i] * dt
            # Push runoff to queue
            for time, value in zip(times, values):
                heappush(queue, (time, value))
            # Add up runoff contributed up to current time step
            while queue:
                time, value = heappop(queue)
                if time > yield_time:
                    heappush(queue, (time, value))
                    break
                result += value
            Q_overflow_t[i] = result
        # TODO: Note that this is not a rate
        self.Q_overflow_t = Q_overflow_t

    def load_model(self, obj, load_optional=True):
        required_fields = {'alpha_fc', 'bb', 'D', 'satdk', 'satpsi', 'slop', 
                           'smcmax', 'smcwlt', 'K_lf', 'giuh_values', 'giuh_timedeltas'}
        optional_fields = set()
        defaults = {}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
        try:
            # TODO: This can be condensed
            assert isinstance(obj['alpha_fc'], np.ndarray)
            assert isinstance(obj['bb'], np.ndarray)
            assert isinstance(obj['D'], np.ndarray)
            assert isinstance(obj['satdk'], np.ndarray)
            assert isinstance(obj['satpsi'], np.ndarray)
            assert isinstance(obj['slop'], np.ndarray)
            assert isinstance(obj['smcmax'], np.ndarray)
            assert isinstance(obj['smcwlt'], np.ndarray)
            assert isinstance(obj['K_lf'], np.ndarray)
            assert isinstance(obj['giuh_values'], list)
            assert isinstance(obj['giuh_timedeltas'], list)
            assert obj['alpha_fc'].dtype == np.float64
            assert obj['bb'].dtype == np.float64
            assert obj['D'].dtype == np.float64
            assert obj['satdk'].dtype == np.float64
            assert obj['satpsi'].dtype == np.float64
            assert obj['slop'].dtype == np.float64
            assert obj['smcmax'].dtype == np.float64
            assert obj['smcwlt'].dtype == np.float64
            assert obj['K_lf'].dtype == np.float64
            #assert obj['giuh_values'].dtype == np.float64
            #assert obj['giuh_timedeltas'].dtype == pd.Timedelta
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            # TODO: This too
            assert (obj['bb'].size == obj['D'].size == obj['satdk'].size ==
                    obj['satpsi'].size == obj['slop'].size == obj['smcmax'].size ==
                    obj['smcwlt'].size == obj['K_lf'].size)
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


class GroundwaterLayer():
    def __init__(self, parent, data):
        self.parent = parent
        self.load_model(data)
        self.S_gw_t = self.S_gw_max * 0.01
        self.Q_gw_t = np.zeros(self.S_gw_t.size, dtype=np.float64)
        self.saved_states = {
            'datetime' : copy.copy(self.datetime),
            'S_gw_t' : self.S_gw_t.copy(),
            'Q_gw_t' : self.Q_gw_t.copy()
        }

    @property
    def datetime(self):
        return self.parent.datetime

    @property
    def timedelta(self):
        return self.parent.timedelta

    @property
    def dt(self):
        return self.parent.timedelta.seconds

    @property
    def Q_perc_t(self):
        return self.parent.soil_layer.Q_perc_t

    def calculate_saturation_excess_overland_flow_from_gw(self):
        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        # TODO: Figure out way to implement this that is consistent with ODE
        pass

    def compute_groundwater_flux__exponential(self):
        C_gw = self.C_gw
        S_gw_t = self.S_gw_t
        k_gw = self.k_gw
        S_gw_max = self.S_gw_max
        Q_gw_t = C_gw * (np.exp(k_gw * S_gw_t / S_gw_max) - 1.)
        self.Q_gw_t = Q_gw_t

    def calculate_groundwater_storage__explicit(self, dt):
        if dt is None:
            dt = self.dt
        Q_perc_t = self.Q_perc_t
        Q_gw_t = self.Q_gw_t
        S_gw_t_prev = self.saved_states['S_gw_t']
        S_gw_t_next = S_gw_t_prev + dt * (Q_perc_t - Q_gw_t)
        self.S_gw_t = S_gw_t_next

    def calculate_groundwater_storage__trapezoidal(self, dt):
        if dt is None:
            dt = self.dt
        Q_perc_t = self.Q_perc_t
        Q_gw_t = self.Q_gw_t
        S_gw_t_prev = self.saved_states['S_gw_t']
        Q_perc_t_prev = self.parent.soil_layer.saved_states['Q_perc_t']
        Q_gw_t_prev = self.saved_states['Q_gw_t']
        f_prev = Q_perc_t_prev - Q_gw_t_prev
        f_next = Q_perc_t - Q_gw_t
        S_gw_t_next = S_gw_t_prev + dt / 2 * (f_prev + f_next)
        self.S_gw_t = S_gw_t_next

    def save_state(self):
        self.saved_states['datetime'] = self.datetime
        self.saved_states['S_gw_t'] = self.S_gw_t.copy()
        self.saved_states['Q_gw_t'] = self.Q_gw_t.copy()

    def load_state(self):
        self.datetime = self.saved_states['datetime']
        self.S_gw_t = self.saved_states['S_gw_t']

    def load_model(self, obj, load_optional=True):
        required_fields = {'S_gw_max', 'C_gw', 'k_gw'}
        optional_fields = set()
        defaults = {}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
        try:
            # TODO: This can be condensed
            assert isinstance(obj['S_gw_max'], np.ndarray)
            assert isinstance(obj['C_gw'], np.ndarray)
            assert isinstance(obj['k_gw'], np.ndarray)
            assert obj['S_gw_max'].dtype == np.float64
            assert obj['C_gw'].dtype == np.float64
            assert obj['k_gw'].dtype == np.float64
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            # TODO: This too
            assert (obj['S_gw_max'].size == obj['C_gw'].size == obj['k_gw'].size)
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

@njit
def compute_et_from_soil(S_t, S_thresh, S_wilt, pet_t):
    n = len(S_t)
    et_soil_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (S_t[i] >= S_thresh[i]):
            et_soil_t[i] = pet_t[i]
        elif (S_t[i] > S_wilt[i]) & (S_t[i] < S_thresh[i]):
            et_soil_t[i] = pet_t[i] * (S_t[i] - S_wilt[i]) / (S_thresh[i] - S_wilt[i])
        elif (S_t[i] <= S_wilt[i]):
            et_soil_t[i] = 0.
        else:
            raise ValueError('Check values of S_wilt and S_thresh')
    return et_soil_t

@njit
def compute_lateral_flow_in_soil(S_t, S_thresh, S_max, K_lf):
    n = len(S_t)
    Q_lf_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (S_t[i] >= S_thresh[i]):
            Q_lf_t[i] = K_lf[i] * (S_t[i] - S_thresh[i]) / (S_max[i] - S_thresh[i])
        elif (S_t[i] < S_thresh[i]):
            Q_lf_t[i] = 0.
        else:
            raise ValueError('Check values of S_t and S_thresh')
    return Q_lf_t

@njit
def compute_percolation_from_soil(S_t, S_thresh, S_max, K_perc):
    n = len(S_t)
    Q_perc_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (S_t[i] >= S_thresh[i]):
            Q_perc_t[i] = K_perc[i] * (S_t[i] - S_thresh[i]) / (S_max[i] - S_thresh[i])
        elif (S_t[i] < S_thresh[i]):
            Q_perc_t[i] = 0.
        else:
            raise ValueError('Check values of S_t and S_thresh')
    return Q_perc_t

@njit
def compute_infiltration_rate__schaake(S_t, p_t, S_max, schaake_constant):
    n = len(S_t)
    I_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        S_deficit = S_max[i] - S_t[i]
        if S_deficit < 0:
            I_t[i] = 0.
        else:
            I_c_t = S_deficit * (1 - math.exp(schaake_constant[i]))
            I_t[i] = min(p_t[i] * I_c_t / (p_t[i] + I_c_t), p_t[i])
    return I_t
        
