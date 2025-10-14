import sys
import math
import uuid
import numpy as np
import pandas as pd
from numba import njit
from scipy.integrate import odeint
from scipy.signal import lsim
import copy
from heapq import heappop, heappush

DEFAULT_START_TIME = pd.to_datetime(0., utc=True)
DEFAULT_TIMEDELTA = pd.to_timedelta(3600, unit='s')

# Adapted from cfe_py code at: https://github.com/NWC-CUAHSI-Summer-Institute/cfe_py

class CFEModel():
    def __init__(self, data):
        self.load_model(data)
        self.N = len(self.watershed_ids)
        self.surface_layer = SurfaceLayer(self, data) 
        self.soil_layer = SoilLayer(self, data) 
        self.groundwater_layer = GroundwaterLayer(self, data)

    @property
    def dt(self):
        return self.timedelta.seconds

    def step(self, p_t, pet_t, dt=None, num_iter=40, eps=1e-9,
             max_learning_rate=0.5, min_learning_rate=0.01):
        self.save_state()
        surface_layer = self.surface_layer
        soil_layer = self.soil_layer
        groundwater_layer = self.groundwater_layer
        S_surf_t_prev = surface_layer.S_surf_t.copy()
        S_t_prev = soil_layer.S_t.copy()
        S_gw_t_prev = groundwater_layer.S_gw_t.copy()
        self.iter_elapsed = 0
        for _ in range(num_iter):
            # Calculate fluxes
            # Rainfall and ET 
            soil_layer.calculate_evaporation_from_rainfall(p_t, pet_t)
            soil_layer.calculate_evaporation_from_soil(pet_t)
            # Infiltration partitioning
            soil_layer.calculate_infiltration_rate(p_t)
            soil_layer.calculate_lateral_flow_in_soil()
            soil_layer.calculate_percolation_from_soil()
            # Surface water reservoir
            surface_layer.calculate_surface_runoff_rate()
            # Groundwater model
            groundwater_layer.calculate_saturation_excess_overland_flow_from_gw()
            groundwater_layer.compute_groundwater_flux__exponential()  

            # Calculate water storages
            # Surface ponding
            S_surf_t_next = surface_layer.calculate_surface_storage__trapezoidal(dt, p_t)
            # Soil moisture reservoir
            S_t_next = soil_layer.calculate_soil_storage__trapezoidal(dt)
            # Groundwater storage
            S_gw_t_next = groundwater_layer.calculate_groundwater_storage__trapezoidal(dt)

            # Set new soil moisture states
            #surf_ratio = S_surf_t_prev / (S_surf_t_next - S_surf_t_prev)
            #soil_ratio = S_t_prev / (S_t_next - S_t_prev)
            #gw_ratio = S_gw_t_prev / (S_gw_t_next - S_gw_t_prev)
            #binding_ratio = min(surf_ratio.min(), soil_ratio.min(), gw_ratio.min())
            #learning_rate = max(min(binding_ratio, max_learning_rate), min_learning_rate)
            learning_rate = 0.25
            self.surface_layer.S_surf_t = (1 - learning_rate) * S_surf_t_prev + (learning_rate) * S_surf_t_next
            self.soil_layer.S_t = (1 - learning_rate) * S_t_prev + (learning_rate) * S_t_next
            self.groundwater_layer.S_gw_t = (1 - learning_rate) * S_gw_t_prev + (learning_rate) * S_gw_t_next

            # Continue iterating until convergence
            self.iter_elapsed += 1
            surf_rel_err = S_surf_t_next - S_surf_t_prev
            soil_rel_err = S_t_next - S_t_prev
            gw_rel_err = S_gw_t_next - S_gw_t_prev
            max_rel_err = max(np.abs(surf_rel_err).max(),
                              np.abs(soil_rel_err).max(),
                              np.abs(gw_rel_err).max())
            if max_rel_err > eps:
                S_surf_t_prev = surface_layer.S_surf_t.copy()
                S_t_prev = soil_layer.S_t.copy()
                S_gw_t_prev = groundwater_layer.S_gw_t.copy()
            else:
                break

        # Compute nash cascade
        self.soil_layer.calculate_nash_cascade__lsim()
        # Compute runoff by convolution with GIUH
        self.surface_layer.calculate_surface_runoff__giuh()
        # Update timestamp
        self.datetime = self.datetime + self.timedelta

    def save_state(self):
        self.surface_layer.save_state()
        self.soil_layer.save_state()
        self.groundwater_layer.save_state()

    def load_state(self):
        self.surface_layer.load_state()
        self.soil_layer.load_state()
        self.groundwater_layer.load_state()

    def load_model(self, obj, load_optional=True):
        required_fields = {'name', 'datetime', 'timedelta',
                           'watershed_ids', 'catchment_area_m2'}
        optional_fields = set()
        defaults = {}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
        try:
            # TODO: This can be condensed
            assert isinstance(obj['watershed_ids'], list)
            assert isinstance(obj['catchment_area_m2'], np.ndarray)
            assert obj['catchment_area_m2'].dtype == np.float64
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            # TODO: This too
            assert (obj['catchment_area_m2'].size == len(obj['watershed_ids']))
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


class SurfaceLayer():
    def __init__(self, parent, data):
        self.parent = parent
        self.load_model(data)
        self.S_surf_t = np.zeros(self.parent.N, dtype=np.float64)
        self.q_surf_t = np.zeros(self.parent.N, dtype=np.float64)
        self.q_overflow_t = np.zeros(self.parent.N, dtype=np.float64)
        self.runoff_queues = [[] for _ in range(self.parent.N)]
        self.saved_states = {
            'datetime' : copy.copy(self.datetime),
            'S_surf_t' : self.S_surf_t.copy(),
            'q_surf_t' : self.q_surf_t.copy()
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
        self.saved_states['S_surf_t'] = self.S_surf_t.copy()
        self.saved_states['q_surf_t'] = self.q_surf_t.copy()
        self.saved_states['q_overflow_t'] = self.q_overflow_t.copy()

    def load_state(self):
        self.datetime = self.saved_states['datetime']
        self.S_surf_t = self.saved_states['S_surf_t']

    def calculate_surface_runoff_rate(self):
        S_surf_t = self.S_surf_t
        n = self.mannings_n
        S_o = self.surf_slope
        B = self.watershed_width
        surf_area = self.parent.catchment_area_m2
        h_t = np.maximum(S_surf_t, 0.)
        q_surf_t = (1 / n) * h_t**(5/3) * B * np.sqrt(S_o) / surf_area
        self.q_surf_t = q_surf_t

    def calculate_surface_storage__explicit(self, dt, p_t):
        if dt is None:
            dt = self.dt
        q_surf_t = self.q_surf_t
        I_t = self.parent.soil_layer.I_t
        S_surf_t_prev = self.saved_states['S_surf_t']
        S_surf_t_next = S_surf_t_prev + dt * (p_t - I_t - q_surf_t)
        self.S_surf_t = S_surf_t_next

    def calculate_surface_storage__trapezoidal(self, dt, p_t):
        if dt is None:
            dt = self.dt
        q_surf_t = self.q_surf_t
        I_t = self.parent.soil_layer.I_t
        S_surf_t_prev = self.saved_states['S_surf_t']
        q_surf_t_prev = self.saved_states['q_surf_t']
        I_t_prev = self.parent.soil_layer.saved_states['I_t']
        f_prev = p_t - I_t_prev - q_surf_t_prev
        f_next = p_t - I_t - q_surf_t
        S_surf_t_next = S_surf_t_prev + dt / 2 * (f_prev + f_next)
        return S_surf_t_next
        #self.S_surf_t = S_surf_t_next

    def calculate_surface_runoff__giuh(self):
        dt = self.dt
        runoff_queues = self.runoff_queues
        yield_time = self.datetime + self.timedelta
        q_surf_t = self.q_surf_t
        giuh_timedeltas = self.giuh_timedeltas
        giuh_values = self.giuh_values
        q_overflow_t = self.q_overflow_t
        for i, queue in enumerate(runoff_queues):
            result = 0.
            times = giuh_timedeltas[i] + yield_time
            values = giuh_values[i] * q_surf_t[i] * dt
            # Push runoff to queue
            for time, value in zip(times, values):
                heappush(queue, (time, value))
            # Add up runoff contributed up to current time step
            min_time = yield_time
            max_time = self.datetime
            while queue:
                time, value = heappop(queue)
                min_time = min(time, min_time)
                max_time = max(time, max_time)
                if time > yield_time:
                    heappush(queue, (time, value))
                    break
                result += value
            time_diff = (max_time - min_time).seconds
            result = result / time_diff
            q_overflow_t[i] = result
        # TODO: Note that this is not a rate
        self.q_overflow_t = q_overflow_t

    def load_model(self, obj, load_optional=True):
        required_fields = {'giuh_values', 'giuh_timedeltas', 'surf_slope', 'mannings_n', 'watershed_width'}
        optional_fields = set()
        defaults = {}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
        try:
            # TODO: This can be condensed
            assert isinstance(obj['giuh_values'], list)
            assert isinstance(obj['giuh_timedeltas'], list)
            assert isinstance(obj['surf_slope'], np.ndarray)
            assert isinstance(obj['mannings_n'], np.ndarray)
            assert isinstance(obj['watershed_width'], np.ndarray)
            assert obj['surf_slope'].dtype == np.float64
            assert obj['mannings_n'].dtype == np.float64
            assert obj['watershed_width'].dtype == np.float64
            #assert obj['giuh_values'].dtype == np.float64
            #assert obj['giuh_timedeltas'].dtype == pd.Timedelta
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            # TODO: This too
            assert (obj['surf_slope'].size == obj['mannings_n'].size == obj['watershed_width'].size)
            assert (len(obj['giuh_values']) == len(obj['giuh_timedeltas']))
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

        self.I_t = np.zeros(self.parent.N, dtype=np.float64)
        self.et_soil_t = np.zeros(self.parent.N, dtype=np.float64)
        self.q_lf_t = np.zeros(self.parent.N, dtype=np.float64)
        self.q_perc_t = np.zeros(self.parent.N, dtype=np.float64)

        # Schaake partitioning
        self.refkdt = 3.0
        self.satdk_ref = 2e-6
        self.schaake_constant = self.refkdt * self.satdk / self.satdk_ref

        # TODO: Check this
        self.K_perc = self.satdk * self.slop

        # Nash cascade
        self.S_nash_t = []
        self.nash_ss = []
        for i in range(self.parent.N):
            num_cascades_i = self.num_nash_cascades[i]
            K_nash_i = self.K_nash[i]
            S_nash_t_i = np.zeros(num_cascades_i, dtype=np.float64)
            self.S_nash_t.append(S_nash_t_i)
            Ks = np.repeat(K_nash_i, num_cascades_i)
            A = np.diag(-Ks) + np.diag(Ks[:-1], k=-1)
            B = np.zeros((Ks.size, 1))
            B[0, 0] = 1.
            C = np.zeros((1, Ks.size))
            C[0, -1] = K_nash_i
            D = np.zeros((1, 1))
            ss_i = (A, B, C, D)
            self.nash_ss.append(ss_i)

        self.q_bucket_t = np.zeros(self.parent.N, dtype=np.float64)

        self.saved_states = {
            'datetime' : copy.copy(self.datetime),
            'S_t' : self.S_t.copy(),
            'I_t' : self.I_t.copy(),
            'et_soil_t' : self.et_soil_t.copy(),
            'q_lf_t' : self.q_lf_t.copy(),
            'q_perc_t' : self.q_perc_t.copy()
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
        self.saved_states['q_lf_t'] = self.q_lf_t.copy()
        self.saved_states['q_perc_t'] = self.q_perc_t.copy()
        self.saved_states['S_nash_t'] = copy.deepcopy(self.S_nash_t)

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
        q_lf_t = compute_lateral_flow_in_soil(S_t, S_thresh, S_max, K_lf)
        self.q_lf_t = q_lf_t

    def calculate_percolation_from_soil(self):
        S_t = self.S_t
        S_thresh = self.S_thresh
        S_max = self.S_max
        K_perc = self.K_perc
        q_perc_t = compute_percolation_from_soil(S_t, S_thresh, S_max, K_perc)
        self.q_perc_t = q_perc_t

    def calculate_infiltration_rate(self, p_t):
        S_t = self.S_t
        S_max = self.S_max
        schaake_constant = self.schaake_constant
        I_t = compute_infiltration_rate__schaake(S_t, p_t, S_max, schaake_constant)
        self.I_t = I_t

    def calculate_soil_storage__explicit(self, dt):
        if dt is None:
            dt = self.dt
        S_t_prev = self.saved_states['S_t']
        I_t = self.I_t
        et_soil_t = self.et_soil_t
        q_lf_t = self.q_lf_t
        q_perc_t = self.q_perc_t
        S_t_next = S_t_prev + dt * (I_t - et_soil_t - q_lf_t - q_perc_t)
        self.S_t = S_t_next

    def calculate_soil_storage__trapezoidal(self, dt):
        if dt is None:
            dt = self.dt
        S_t_prev = self.saved_states['S_t']
        I_t_prev = self.saved_states['I_t']
        et_soil_t_prev = self.saved_states['et_soil_t']
        q_lf_t_prev = self.saved_states['q_lf_t']
        q_perc_t_prev = self.saved_states['q_perc_t']
        I_t = self.I_t
        et_soil_t = self.et_soil_t
        q_lf_t = self.q_lf_t
        q_perc_t = self.q_perc_t
        f_prev = (I_t_prev - et_soil_t_prev - q_lf_t_prev - q_perc_t_prev)
        f_next = (I_t - et_soil_t - q_lf_t - q_perc_t)
        S_t_next = S_t_prev + dt / 2 * (f_prev + f_next)
        return S_t_next
        #self.S_t = S_t_next

    def calculate_nash_cascade__sequential(self):
        dt = self.dt
        S_nash_t = self.S_nash_t
        K_nash = self.K_nash
        q_lf_t = np.maximum(self.q_lf_t, 0.)
        q_bucket_t = self.q_bucket_t
        for i, storage in enumerate(S_nash_t):
            num_cascades = len(storage)
            storage[0] += q_lf_t[i] * dt
            if num_cascades > 1:
                for j in range(1, len(storage)):
                    q_cascade = max(K_nash[i] * storage[j-1], 0.)
                    storage[j] += q_cascade * dt
                    storage[j-1] -= q_cascade * dt
            q_out = max(K_nash[i] * storage[-1], 0.)
            q_bucket_t[i] = q_out
            storage[-1] -= q_out * dt
        self.q_bucket_t = q_bucket_t

    def calculate_nash_cascade__lsim(self):
        dt = self.dt
        S_nash_t = self.S_nash_t
        S_nash_t_prev = self.saved_states['S_nash_t']
        q_lf_t = self.q_lf_t
        q_lf_t_prev = self.saved_states['q_lf_t']
        nash_ss = self.nash_ss
        q_bucket_t = self.q_bucket_t
        for i in range(self.parent.N):
            S_nash_t_prev_i = S_nash_t_prev[i]
            ss_i = nash_ss[i]
            q_lf_t_i = q_lf_t[i]
            q_lf_t_prev_i = q_lf_t_prev[i]
            t, y, x = lsim(ss_i, U=[q_lf_t_prev_i, q_lf_t_i], 
                           T=[0, dt], 
                           X0=S_nash_t_prev_i)
            S_nash_t[i] = x[-1]
            q_bucket_t[i] = y[-1]
        self.S_nash_t = S_nash_t
        self.q_bucket_t = q_bucket_t

    def load_model(self, obj, load_optional=True):
        required_fields = {'alpha_fc', 'bb', 'D', 'satdk', 'satpsi', 'slop', 
                           'smcmax', 'smcwlt', 'K_lf', 'giuh_values', 'giuh_timedeltas', 
                           'K_nash', 'num_nash_cascades'}
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
            assert isinstance(obj['K_nash'], np.ndarray)
            assert isinstance(obj['num_nash_cascades'], np.ndarray)
            assert obj['alpha_fc'].dtype == np.float64
            assert obj['bb'].dtype == np.float64
            assert obj['D'].dtype == np.float64
            assert obj['satdk'].dtype == np.float64
            assert obj['satpsi'].dtype == np.float64
            assert obj['slop'].dtype == np.float64
            assert obj['smcmax'].dtype == np.float64
            assert obj['smcwlt'].dtype == np.float64
            assert obj['K_lf'].dtype == np.float64
            assert obj['K_nash'].dtype == np.float64
            assert obj['num_nash_cascades'].dtype == np.int64
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            # TODO: This too
            assert (obj['bb'].size == obj['D'].size == obj['satdk'].size ==
                    obj['satpsi'].size == obj['slop'].size == obj['smcmax'].size ==
                    obj['smcwlt'].size == obj['K_lf'].size == obj['K_nash'].size ==
                    obj['num_nash_cascades'].size)
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
        # TODO: Arbitrary instantiation
        self.S_gw_t = self.S_gw_max * 0.01
        self.q_gw_t = np.zeros(self.parent.N, dtype=np.float64)
        self.saved_states = {
            'datetime' : copy.copy(self.datetime),
            'S_gw_t' : self.S_gw_t.copy(),
            'q_gw_t' : self.q_gw_t.copy()
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
    def q_perc_t(self):
        return self.parent.soil_layer.q_perc_t

    def calculate_saturation_excess_overland_flow_from_gw(self):
        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        # TODO: Figure out way to implement this that is consistent with ODE
        pass

    def compute_groundwater_flux__exponential(self):
        C_gw = self.C_gw
        S_gw_t = self.S_gw_t
        k_gw = self.k_gw
        S_gw_max = self.S_gw_max
        q_gw_t = C_gw * (np.exp(k_gw * S_gw_t / S_gw_max) - 1.)
        self.q_gw_t = q_gw_t

    def calculate_groundwater_storage__explicit(self, dt):
        if dt is None:
            dt = self.dt
        q_perc_t = self.q_perc_t
        q_gw_t = self.q_gw_t
        S_gw_t_prev = self.saved_states['S_gw_t']
        S_gw_t_next = S_gw_t_prev + dt * (q_perc_t - q_gw_t)
        self.S_gw_t = S_gw_t_next

    def calculate_groundwater_storage__trapezoidal(self, dt):
        if dt is None:
            dt = self.dt
        q_perc_t = self.q_perc_t
        q_gw_t = self.q_gw_t
        S_gw_t_prev = self.saved_states['S_gw_t']
        q_perc_t_prev = self.parent.soil_layer.saved_states['q_perc_t']
        q_gw_t_prev = self.saved_states['q_gw_t']
        f_prev = q_perc_t_prev - q_gw_t_prev
        f_next = q_perc_t - q_gw_t
        S_gw_t_next = S_gw_t_prev + dt / 2 * (f_prev + f_next)
        return S_gw_t_next
        #self.S_gw_t = S_gw_t_next

    def save_state(self):
        self.saved_states['datetime'] = self.datetime
        self.saved_states['S_gw_t'] = self.S_gw_t.copy()
        self.saved_states['q_gw_t'] = self.q_gw_t.copy()

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
    q_lf_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (S_t[i] >= S_thresh[i]):
            q_lf_t[i] = K_lf[i] * (S_t[i] - S_thresh[i]) / (S_max[i] - S_thresh[i])
        elif (S_t[i] < S_thresh[i]):
            q_lf_t[i] = 0.
        else:
            raise ValueError('Check values of S_t and S_thresh')
    return q_lf_t

@njit
def compute_percolation_from_soil(S_t, S_thresh, S_max, K_perc):
    n = len(S_t)
    q_perc_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if (S_t[i] >= S_thresh[i]):
            q_perc_t[i] = K_perc[i] * (S_t[i] - S_thresh[i]) / (S_max[i] - S_thresh[i])
        elif (S_t[i] < S_thresh[i]):
            q_perc_t[i] = 0.
        else:
            raise ValueError('Check values of S_t and S_thresh')
    return q_perc_t

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
        
