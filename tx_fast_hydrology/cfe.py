import sys
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

class CFE():
    def __init__(self):
        # these need to be initialized here as scale_output() called in update()
        self.streamflow_cmh = 0.0
        self.surface_runoff_m = 0.0

        data_loaded = {
            "forcing_file":"./cat58_01Dec2015.csv",
            "catchment_area_km2":15.617167355002097,
            "alpha_fc":0.33,
            "partition_scheme":"Schaake",
            "soil_params":{
                        "bb":4.05,
                        "satdk":0.00000338,
                        "satpsi":0.355,
                        "slop":1.0,
                        "smcmax":0.439,
                        "wltsmc":0.066,
                        "D":2.0
                        },
            "max_gw_storage":16.0,
            "Cgw":0.01,
            "expon":6.0,
            "gw_storage":0.50,
            "soil_storage":0.667,
            "K_lf":0.01,
            "K_nash":0.03,
            "nash_storage":[0.0,0.0],
            "giuh_ordinates":[0.1,0.35,0.2,0.14,0.1,0.06,0.05],
            "stand_alone":1,
            "unit_test":1,
            "compare_results_file":"cat58_test_compare.csv",
            "soil_scheme":"ode"
        }

        self.forcing_file = data_loaded["forcing_file"]
        self.catchment_area_km2 = data_loaded["catchment_area_km2"]

        # Soil parameters
        self.alpha_fc = data_loaded["alpha_fc"]
        self.soil_params = {}
        self.soil_params["bb"] = data_loaded["soil_params"]["bb"]
        self.soil_params["D"] = data_loaded["soil_params"]["D"]
        self.soil_params["satdk"] = data_loaded["soil_params"]["satdk"]
        self.soil_params["satpsi"] = data_loaded["soil_params"]["satpsi"]
        self.soil_params["slop"] = data_loaded["soil_params"]["slop"]
        self.soil_params["smcmax"] = data_loaded["soil_params"]["smcmax"]
        self.soil_params["wltsmc"] = data_loaded["soil_params"]["wltsmc"]
        self.K_lf = data_loaded["K_lf"]
        self.soil_params["scheme"] = data_loaded["soil_scheme"]

        # Groundwater parameters
        self.max_gw_storage = data_loaded["max_gw_storage"]
        self.Cgw = data_loaded["Cgw"]
        self.expon = data_loaded["expon"]

        # Other modules
        self.K_nash = data_loaded["K_nash"]
        self.nash_storage = np.array(data_loaded["nash_storage"])
        self.giuh_ordinates = np.array(data_loaded["giuh_ordinates"])

        # Partitioning parameters
        self.surface_partitioning_scheme = data_loaded["partition_scheme"]

        # OPTIONAL CONFIGURATIONS
        if "stand_alone" in data_loaded.keys():
            self.stand_alone = data_loaded["stand_alone"]
        if "forcing_file" in data_loaded.keys():
            self.reads_own_forcing = True
            self.forcing_file = data_loaded["forcing_file"]
        if "unit_test" in data_loaded.keys():
            self.unit_test = data_loaded["unit_test"]
            self.compare_results_file = data_loaded["compare_results_file"]
        # Soil representation selection
        if "soil_scheme" in data_loaded.keys():
            self.soil_scheme = data_loaded["soil_scheme"]
        else:
            self.soil_scheme = "classic"

        self.volstart = 0
        self.vol_et_from_soil = 0
        self.vol_et_from_rain = 0
        self.vol_partition_runoff = 0
        self.vol_partition_infilt = 0
        self.vol_out_giuh = 0
        self.vol_end_giuh = 0
        self.vol_to_gw = 0
        self.vol_to_gw_start = 0
        self.vol_to_gw_end = 0
        self.vol_from_gw = 0
        self.vol_in_nash = 0
        self.vol_in_nash_end = 0
        self.vol_out_nash = 0
        self.vol_soil_start = 0
        self.vol_to_soil = 0
        self.vol_soil_to_lat_flow = 0
        self.vol_soil_to_gw = 0
        self.vol_soil_end = 0
        self.volin = 0
        self.volout = 0
        self.volend = 0
        # Added by Ryoko for soil-ode
        self.vol_partition_runoff_IOF = 0
        self.vol_partition_runoff_SOF = 0
        self.vol_et_to_atm = 0
        self.vol_et_from_soil = 0
        self.vol_et_from_rain = 0
        self.vol_PET = 0

        # initialize simulation constants
        atm_press_Pa = 101325.0
        unit_weight_water_N_per_m3 = 9810.0

        # Time control
        self.time_step_size = 3600
        self.timestep_h = self.time_step_size / 3600
        self.timestep_d = self.timestep_h / 24.0
        self.current_time_step = 0
        self.current_time = pd.Timestamp(year=2007, month=10, day=1, hour=0)

        # Inputs
        self.timestep_rainfall_input_m = 0
        self.potential_et_m_per_s = 0

        # calculated flux variables
        self.flux_overland_m = (
            0  # surface runoff that goes through the GIUH convolution process
        )
        self.flux_perc_m = 0  # flux from soil to deeper groundwater reservoir
        self.flux_lat_m = 0  # lateral flux in the subsurface to the Nash cascade
        self.flux_from_deep_gw_to_chan_m = (
            0  # flux from the deep reservoir into the channels
        )
        self.gw_reservoir_storage_deficit_m = (
            0  # the available space in the conceptual groundwater reservoir
        )
        self.primary_flux = 0  # temporary vars.
        self.secondary_flux = 0  # temporary vars.
        self.total_discharge = 0
        # Added by Ryoko for soil-ode
        self.diff_infilt = 0
        self.diff_perc = 0
        # Evapotranspiration
        self.potential_et_m_per_timestep = 0
        self.actual_et_m_per_timestep = 0
        # Added by Ryoko for soil-ode
        self.reduced_potential_et_m_per_timestep = 0
        self.actual_et_from_rain_m_per_timestep = 0
        self.actual_et_from_soil_m_per_timestep = 0
        # Set these values now that we have the information from the configuration file.
        self.runoff_queue_m_per_timestep = np.zeros(len(self.giuh_ordinates) + 1)
        self.num_giuh_ordinates = len(self.giuh_ordinates)
        self.num_lateral_flow_nash_reservoirs = self.nash_storage.shape[0]

        # Local values to be used in setting up soil reservoir
        trigger_z_m = 0.5
        field_capacity_atm_press_fraction = self.alpha_fc

        # SOIL RESERVOIR CONFIGURATION
        # Soil outflux calculation, Equation 3 in Fred Ogden's document
        H_water_table_m = (
            field_capacity_atm_press_fraction
            * atm_press_Pa
            / unit_weight_water_N_per_m3
        )

        soil_water_content_at_field_capacity = self.soil_params["smcmax"] * np.power(
            H_water_table_m / self.soil_params["satpsi"], (1.0 / self.soil_params["bb"])
        )

        Omega = H_water_table_m - trigger_z_m

        # Upper & lower limit of the integral in Equation 4 in Fred Ogden's document
        lower_lim = np.power(Omega, (1.0 - 1.0 / self.soil_params["bb"])) / (
            1.0 - 1.0 / self.soil_params["bb"]
        )

        upper_lim = np.power(
            Omega + self.soil_params["D"], (1.0 - 1.0 / self.soil_params["bb"])
        ) / (1.0 - 1.0 / self.soil_params["bb"])

        # Integral & power term in Equation 4 & 5 in Fred Ogden's document
        storage_thresh_pow_term = np.power(
            1.0 / self.soil_params["satpsi"], (-1.0 / self.soil_params["bb"])
        )

        lim_diff = upper_lim - lower_lim

        field_capacity_storage_threshold_m = (
            self.soil_params["smcmax"] * storage_thresh_pow_term * lim_diff
        )

        # lateral flow function parameters
        assumed_near_channel_water_table_slope = 0.01  # [L/L]
        lateral_flow_threshold_storage_m = field_capacity_storage_threshold_m  # Equation 4 & 5  in Fred Ogden's document
        #         lateral_flow_linear_reservoir_constant = 2.0 * assumed_near_channel_water_table_slope * \     # Not used
        #                                                  self.soil_params['mult'] * NWM_soil_params.satdk * \ # Not used
        #                                                  self.soil_params['D'] * drainage_density_km_per_km2  # Not used
        #         lateral_flow_linear_reservoir_constant *= 3600.0                                              # Not used
        self.soil_reservoir_storage_deficit_m = 0

        # ________________________________________________
        # Subsurface reservoirs
        self.gw_reservoir = {
            "is_exponential": True,
            "storage_max_m": self.max_gw_storage,
            "coeff_primary": self.Cgw,
            "exponent_primary": self.expon,
            "storage_threshold_primary_m": 0.0,
            # The following parameters don't matter. Currently one storage is default. The secoundary storage is turned off.
            "storage_threshold_secondary_m": 0.0,
            "coeff_secondary": 0.0,
            "exponent_secondary": 1.0,
        }
        self.gw_reservoir["storage_m"] = self.gw_reservoir["storage_max_m"] * 0.01
        self.volstart += self.gw_reservoir["storage_m"]
        self.vol_in_gw_start = self.gw_reservoir["storage_m"]

        self.soil_reservoir = {
            "is_exponential": False,
            "wilting_point_m": self.soil_params["wltsmc"] * self.soil_params["D"],
            "storage_max_m": self.soil_params["smcmax"] * self.soil_params["D"],
            "coeff_primary": self.soil_params["satdk"]
            * self.soil_params["slop"]
            * self.time_step_size,  # Controls percolation to GW, Equation 11
            "exponent_primary": 1.0,  # Controls percolation to GW, FIXED to 1 based on Equation 11
            "storage_threshold_primary_m": field_capacity_storage_threshold_m,
            "coeff_secondary": self.K_lf,  # Controls lateral flow
            "exponent_secondary": 1.0,  # Controls lateral flow, FIXED to 1 based on the Fred Ogden's document
            "storage_threshold_secondary_m": lateral_flow_threshold_storage_m,
        }
        self.soil_reservoir["storage_m"] = self.soil_reservoir["storage_max_m"] * 0.667
        self.volstart += self.soil_reservoir["storage_m"]
        self.vol_soil_start = self.soil_reservoir["storage_m"]

        # Schaake partitioning
        self.refkdt = 3.0
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            self.refkdt * self.soil_params["satdk"] / 2.0e-06
        )
        self.Schaake_output_runoff_m = 0
        self.infiltration_depth_m = 0

        # Nash cascade
        self.K_nash = 0.03  # Default value, but should be set in configuration file

        # The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s
        self.output_factor_cms = (
            (1 / 1000) * (self.catchment_area_km2 * 1000 * 1000) * (1 / 3600)
        )

    def calculate_input_rainfall_and_PET(self, cfe_state):
        """
        Calculate input rainfall and PET 
        """
        # NOTE: This will evaluate to zero because potential ET is zero
        cfe_state.potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        cfe_state.reduced_potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        
    def calculate_evaporation_from_rainfall(self, cfe_state):
        """
        Calculate evaporation from rainfall. If it is raining, take PET from rainfall
        """
        cfe_state.actual_et_from_rain_m_per_timestep = 0
        if(cfe_state.timestep_rainfall_input_m > 0):
            self.et_from_rainfall(cfe_state)
        # NOTE: This part is just volume accounting
        cfe_state.vol_et_from_rain += cfe_state.actual_et_from_rain_m_per_timestep
        cfe_state.vol_et_to_atm += cfe_state.actual_et_from_rain_m_per_timestep
        cfe_state.volout += cfe_state.actual_et_from_rain_m_per_timestep
        
        cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_rain_m_per_timestep
        
    def calculate_evaporation_from_soil(self, cfe_state):
        """
        If the soil moisture calculation scheme is 'classic', calculate the evaporation from the soil
        Elseif the soil moisture calculation scheme is 'ode', do nothing, because evaporation from the soil will be calculated within run_soil_moisture_scheme
        """
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            
            cfe_state.actual_et_from_soil_m_per_timestep = 0
            # If the soil moisture storage is more than wilting point, calculate ET from soil
            if(cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']): 
                self.et_from_soil(cfe_state)

            cfe_state.vol_et_from_soil += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.vol_et_to_atm += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep 

            cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_soil_m_per_timestep
            
        elif cfe_state.soil_params['scheme'].lower() == 'ode':
            None
        
    def calculate_the_soil_moisture_deficit(self, cfe_state):
        """ Calculate the soil moisture deficit
        """
        cfe_state.soil_reservoir_storage_deficit_m = (cfe_state.soil_params['smcmax'] * cfe_state.soil_params['D'] - \
                                                        cfe_state.soil_reservoir['storage_m'])
        
    def calculate_infiltration_excess_overland_flow(self, cfe_state):
        """ Calculates infiltration excess overland flow 
        by running the partitioning scheme based on the choice set in the Configuration file
        """
        if (cfe_state.timestep_rainfall_input_m > 0.0): 
            if cfe_state.surface_partitioning_scheme == "Schaake": 
                self.Schaake_partitioning_scheme(cfe_state)
            elif cfe_state.surface_partitioning_scheme == "Xinanjiang": 
                self.Xinanjiang_partitioning_scheme(cfe_state)
            else: 
                print("Problem: must specify one of Schaake of Xinanjiang partitioning scheme.\n")
                print("Program terminating.:( \n");
                sys.exit(1)
        else: 
            cfe_state.surface_runoff_depth_m = 0.0
            cfe_state.infiltration_depth_m = 0.0

    def calculate_saturation_excess_overland_flow_from_soil(self, cfe_state):
        """ Calculates saturation excess overland flow (SOF)
        This should be run after calculate_infiltration_excess_overland_flow, then, 
        infiltration_depth_m and surface_runoff_depth_m get finalized 
        """
        # If the infiltration is more than the soil moisture deficit, 
        # additional runoff (SOF) occurs and soil get saturated
        if cfe_state.soil_reservoir_storage_deficit_m < cfe_state.infiltration_depth_m:
            diff = cfe_state.infiltration_depth_m - cfe_state.soil_reservoir_storage_deficit_m
            cfe_state.surface_runoff_depth_m += diff
            cfe_state.infiltration_depth_m -= diff
            cfe_state.soil_reservoir_storage_deficit_m = 0
        else:
            # If the infiltration is less than the soil moisture deficit,
            # Infiltration & runoff flux is as calculated in calculate_infiltration_excess_overland_flow()
            None

    def track_infiltration_and_runoff(self, cfe_state):
        """ Tracking runoff & infiltraiton volume with final infiltration & runoff values
        """
        cfe_state.vol_partition_runoff += cfe_state.surface_runoff_depth_m
        cfe_state.vol_partition_infilt += cfe_state.infiltration_depth_m
        cfe_state.vol_to_soil += cfe_state.infiltration_depth_m
        
    def run_soil_moisture_scheme(self, cfe_state):
        """ Run the soil moisture scheme based on the choice set in the Configuration file
        """
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            # Add infiltration flux and calculate the reservoir flux 
            cfe_state.soil_reservoir['storage_m'] += cfe_state.infiltration_depth_m
            self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.soil_reservoir)
        elif cfe_state.soil_params['scheme'].lower() == 'ode':
            # Infiltration flux is added witin the ODE scheme
            self.soil_moisture_flux_calc_with_ode(cfe_state, cfe_state.soil_reservoir)

    def update_outflux_from_soil(self, cfe_state):
        cfe_state.flux_perc_m = cfe_state.primary_flux_m  #percolation_flux
        cfe_state.flux_lat_m = cfe_state.secondary_flux_m # lateral_flux
        
        # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
        # If ODE, outfluxes are already subtracted from the soil moisture storage
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_perc_m
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_lat_m
        
        # If ODE, track actual ET from soil
        if cfe_state.soil_params['scheme'].lower() == 'ode':
            cfe_state.vol_et_from_soil += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.vol_et_to_atm += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_soil_m_per_timestep
            
        elif cfe_state.soil_params['scheme'].lower() == 'classic':
            None

    def calculate_groundwater_storage_deficit(self, cfe_state):
        cfe_state.gw_reservoir_storage_deficit_m = cfe_state.gw_reservoir['storage_max_m'] - cfe_state.gw_reservoir['storage_m']
        
    def calculate_saturation_excess_overland_flow_from_gw(self, cfe_state):
        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        if cfe_state.flux_perc_m > cfe_state.gw_reservoir_storage_deficit_m:
            diff = cfe_state.flux_perc_m - cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.surface_runoff_depth_m += diff
            cfe_state.flux_perc_m = cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.gw_reservoir['storage_m'] = cfe_state.gw_reservoir['storage_max_m']
            cfe_state.gw_reservoir_storage_deficit_m = 0
            cfe_state.vol_partition_runoff += diff 
            cfe_state.vol_partition_infilt -= diff
            
        cfe_state.gw_reservoir['storage_m']   += cfe_state.flux_perc_m
            
    def track_volume_from_percolation_and_lateral_flow(self, cfe_state):
        # Finalize the percolation and lateral flow 
        cfe_state.vol_to_gw                += cfe_state.flux_perc_m
        cfe_state.vol_soil_to_gw           += cfe_state.flux_perc_m
        cfe_state.vol_soil_to_lat_flow     += cfe_state.flux_lat_m  #TODO add this to nash cascade as input
        cfe_state.volout                   += cfe_state.flux_lat_m
    
    def set_flux_from_deep_gw_to_chan_m(self, cfe_state):
        cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.primary_flux_m
        if (cfe_state.flux_from_deep_gw_to_chan_m > cfe_state.gw_reservoir['storage_m']): 
            cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.gw_reservoir['storage_m']
            if cfe_state.verbose:
                print("WARNING: Groundwater flux larger than storage. \n")

        cfe_state.vol_from_gw += cfe_state.flux_from_deep_gw_to_chan_m
        
    def remove_flux_from_deep_gw_to_chan_m(self, cfe_state):
        """ Just an accounting operation
        """
        cfe_state.gw_reservoir['storage_m'] -= cfe_state.flux_from_deep_gw_to_chan_m

    def track_volume_from_giuh(self, cfe_state):
        cfe_state.vol_out_giuh += cfe_state.flux_giuh_runoff_m
        cfe_state.volout += cfe_state.flux_giuh_runoff_m

    def track_volume_from_deep_gw_to_chan(self, cfe_state):
        cfe_state.volout += cfe_state.flux_from_deep_gw_to_chan_m 

    def track_volume_from_nash_cascade(self, cfe_state):
        cfe_state.vol_in_nash += cfe_state.flux_lat_m
        cfe_state.vol_out_nash += cfe_state.flux_nash_lateral_runoff_m

    def add_up_total_flux_discharge(self, cfe_state):
        cfe_state.flux_Qout_m = cfe_state.flux_giuh_runoff_m + cfe_state.flux_nash_lateral_runoff_m + cfe_state.flux_from_deep_gw_to_chan_m
        cfe_state.total_discharge = cfe_state.flux_Qout_m * cfe_state.catchment_area_km2 * 1000000.0 / cfe_state.time_step_size

    def update_current_time(self, cfe_state):
        cfe_state.current_time_step += 1
        cfe_state.current_time      += pd.Timedelta(value=cfe_state.time_step_size, unit='s')

    def nash_cascade(self,cfe_state):
        """
            Solve for the flow through the Nash cascade to delay the 
            arrival of the lateral flow into the channel
        """
        Q = np.zeros(cfe_state.num_lateral_flow_nash_reservoirs)
        
        for i in range(cfe_state.num_lateral_flow_nash_reservoirs):
            
            Q[i] = cfe_state.K_nash * cfe_state.nash_storage[i]
            
            cfe_state.nash_storage[i] -= Q[i]
            
            if i == 0:
                
                cfe_state.nash_storage[i] += cfe_state.flux_lat_m
                
            else:
                
                cfe_state.nash_storage[i] += Q[i-1]
        
        cfe_state.flux_nash_lateral_runoff_m = Q[cfe_state.num_lateral_flow_nash_reservoirs - 1]
        
        return
    

    def convolution_integral(self,cfe_state):
        """
            This function solves the convolution integral involving N GIUH ordinates.
            
            Inputs:
                Schaake_output_runoff_m
                num_giuh_ordinates
                giuh_ordinates
            Outputs:
                runoff_queue_m_per_timestep
        """

        N=cfe_state.num_giuh_ordinates

        cfe_state.runoff_queue_m_per_timestep[N] = 0.0
        
        
        for i in range(cfe_state.num_giuh_ordinates): 

            cfe_state.runoff_queue_m_per_timestep[i] += cfe_state.giuh_ordinates[i] * cfe_state.surface_runoff_depth_m
            
        cfe_state.flux_giuh_runoff_m = cfe_state.runoff_queue_m_per_timestep[0]
        
        # shift all the entries in preperation for the next timestep

        for i in range(cfe_state.num_giuh_ordinates): 
            cfe_state.runoff_queue_m_per_timestep[i] = cfe_state.runoff_queue_m_per_timestep[i+1]

        return
    

    def et_from_rainfall(self,cfe_state):
        
        """
            iff it is raining, take PET from rainfall first.  Wet veg. is efficient evaporator.
        """

        # If rainfall exceeds PET, actual AET from rainfall is equal to the PET
        if cfe_state.timestep_rainfall_input_m > cfe_state.potential_et_m_per_timestep:
            cfe_state.actual_et_from_rain_m_per_timestep = cfe_state.potential_et_m_per_timestep
            cfe_state.timestep_rainfall_input_m -= cfe_state.actual_et_from_rain_m_per_timestep

        # If rainfall is less than PET, all rainfall gets consumed as AET
        else: 
            cfe_state.actual_et_from_rain_m_per_timestep = cfe_state.timestep_rainfall_input_m
            cfe_state.timestep_rainfall_input_m = 0.0
    
        cfe_state.reduced_potential_et_m_per_timestep = cfe_state.potential_et_m_per_timestep - cfe_state.actual_et_from_rain_m_per_timestep
        
        return
                
    ########## SINGLE OUTLET EXPONENTIAL RESERVOIR ###############
    ##########                -or-                 ###############
    ##########    TWO OUTLET NONLINEAR RESERVOIR   ###############                        
    def conceptual_reservoir_flux_calc(self,cfe_state,reservoir):
        """
            This function calculates the flux from a linear, or nonlinear 
            conceptual reservoir with one or two outlets, or from an
            exponential nonlinear conceptual reservoir with only one outlet.
            In the non-exponential instance, each outlet can have its own
            activation storage threshold.  Flow from the second outlet is 
            turned off by setting the discharge coeff. to 0.0.
        """

        if reservoir['is_exponential'] == True: 
            flux_exponential = np.exp(reservoir['exponent_primary'] * \
                                      reservoir['storage_m'] / \
                                      reservoir['storage_max_m']) - 1.0
            cfe_state.primary_flux_m = reservoir['coeff_primary'] * flux_exponential
            cfe_state.secondary_flux_m=0.0
            return
    
        cfe_state.primary_flux_m=0.0
        
        storage_above_threshold_m = reservoir['storage_m'] - reservoir['storage_threshold_primary_m']
        
        if storage_above_threshold_m > 0.0:
                               
            storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = np.power(storage_ratio, reservoir['exponent_primary'])
            
            cfe_state.primary_flux_m = reservoir['coeff_primary'] * storage_power

            if cfe_state.primary_flux_m > storage_above_threshold_m:
                cfe_state.primary_flux_m = storage_above_threshold_m
                
        cfe_state.secondary_flux_m = 0.0
            
        storage_above_threshold_m = reservoir['storage_m'] - reservoir['storage_threshold_secondary_m']
        
        if storage_above_threshold_m > 0.0:
            
            storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_secondary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = np.power(storage_ratio, reservoir['exponent_secondary'])
            
            cfe_state.secondary_flux_m = reservoir['coeff_secondary'] * storage_power
            
            if cfe_state.secondary_flux_m > (storage_above_threshold_m - cfe_state.primary_flux_m):
                cfe_state.secondary_flux_m = storage_above_threshold_m - cfe_state.primary_flux_m
                
        return
    
    
    #  SCHAAKE RUNOFF PARTITIONING SCHEME
    def Schaake_partitioning_scheme(self,cfe_state):
        """
            This subtroutine takes water_input_depth_m and partitions it into surface_runoff_depth_m and
            infiltration_depth_m using the scheme from Schaake et al. 1996. 
            !--------------------------------------------------------------------------------
            modified by FLO April 2020 to eliminate reference to ice processes, 
            and to de-obfuscate and use descriptive and dimensionally consistent variable names.
            
            inputs:
              timestep_d
              Schaake_adjusted_magic_constant_by_soil_type = C*Ks(soiltype)/Ks_ref, where C=3, and Ks_ref=2.0E-06 m/s
              column_total_soil_moisture_deficit_m (soil_reservoir_storage_deficit_m)
              water_input_depth_m (timestep_rainfall_input_m) amount of water input to soil surface this time step [m]
            outputs:
              surface_runoff_depth_m      amount of water partitioned to surface water this time step [m]
              infiltration_depth_m
        """
        
        if 0 < cfe_state.timestep_rainfall_input_m:
            
            if 0 > cfe_state.soil_reservoir_storage_deficit_m:
                
                cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = 0.0
                
            else:
                
                schaake_exp_term = np.exp( - cfe_state.Schaake_adjusted_magic_constant_by_soil_type * cfe_state.timestep_d)
                
                Schaake_parenthetical_term = (1.0 - schaake_exp_term)
                
                Ic = cfe_state.soil_reservoir_storage_deficit_m * Schaake_parenthetical_term
                
                Px = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = (Px * (Ic / (Px + Ic)))
                
                if 0.0 < (cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m):
                    
                    cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m
                    
                else:
                    
                    cfe_state.surface_runoff_depth_m = 0.0
                    
                cfe_state.infiltration_depth_m =  cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m
                    
        else:
            
            cfe_state.surface_runoff_depth_m = 0.0
            cfe_state.infiltration_depth_m = 0.0
            
        return
    
    def Xinanjiang_partitioning_scheme(self,cfe_state): 
        """
            This module takes the water_input_depth_m and separates it into surface_runoff_depth_m
            and infiltration_depth_m by calculating the saturated area and runoff based on a scheme developed
            for the Xinanjiang model by Jaywardena and Zhou (2000). According to Knoben et al.
            (2019) "the model uses a variable contributing area to simulate runoff.  [It] uses
            a double parabolic curve to simulate tension water capacities within the catchment, 
            instead of the original single parabolic curve" which is also used as the standard 
            VIC fomulation.  This runoff scheme was selected for implementation into NWM v3.0.
            REFERENCES:
            1. Jaywardena, A.W. and M.C. Zhou, 2000. A modified spatial soil moisture storage 
                capacity distribution curve for the Xinanjiang model. Journal of Hydrology 227: 93-113
            2. Knoben, W.J.M. et al., 2019. Supplement of Modular Assessment of Rainfall-Runoff Models
                Toolbox (MARRMoT) v1.2: an open-source, extendable framework providing implementations
                of 46 conceptual hydrologic models as continuous state-space formulations. Supplement of 
                Geosci. Model Dev. 12: 2463-2480.
            -------------------------------------------------------------------------
            Written by RLM May 2021
            Adapted by JMFrame September 2021 for new version of CFE
            Further adapted by QiyueL August 2022 for python version of CFE
            ------------------------------------------------------------------------
            Inputs
            double  time_step_rainfall_input_m           amount of water input to soil surface this time step [m]
            double  field_capacity_m                     amount of water stored in soil reservoir when at field capacity [m]
            double  max_soil_moisture_storage_m          total storage of the soil moisture reservoir (porosity*soil thickness) [m]
            double  column_total_soil_water_m     current storage of the soil moisture reservoir [m]
            double  a_inflection_point_parameter  a parameter
            double  b_shape_parameter             b parameter
            double  x_shape_parameter             x parameter
                //
            Outputs
            double  surface_runoff_depth_m        amount of water partitioned to surface water this time step [m]
            double  infiltration_depth_m          amount of water partitioned as infiltration (soil water input) this time step [m]
            ------------------------------------------------------------------------- 
        """

        # partition the total soil water in the column between free water and tension water
        free_water_m = cfe_state.soil_reservoir['storage_m']- cfe_state.soil_reservoir['storage_threshold_primary_m'];

        if (0.0 < free_water_m):

            tension_water_m = cfe_state.soil_reservoir['storage_threshold_primary_m'];

        else: 

            free_water_m = 0.0;
            tension_water_m = cfe_state.soil_reservoir['storage_m']
        
        # estimate the maximum free water and tension water available in the soil column
        max_free_water_m = cfe_state.soil_reservoir['storage_max_m'] - cfe_state.soil_reservoir['storage_threshold_primary_m']
        max_tension_water_m = cfe_state.soil_reservoir['storage_threshold_primary_m']

        # check that the free_water_m and tension_water_m do not exceed the maximum and if so, change to the max value
        if(max_free_water_m < free_water_m): 
            free_water_m = max_free_water_m

        if(max_tension_water_m < tension_water_m): 
            tension_water_m = max_tension_water_m

        """
            NOTE: the impervious surface runoff assumptions due to frozen soil used in NWM 3.0 have not been included.
            We are assuming an impervious area due to frozen soils equal to 0 (see eq. 309 from Knoben et al).

            The total (pervious) runoff is first estimated before partitioning into surface and subsurface components.
            See Knoben et al eq 310 for total runoff and eqs 313-315 for partitioning between surface and subsurface
            components.

            Calculate total estimated pervious runoff. 
            NOTE: If the impervious surface runoff due to frozen soils is added,
            the pervious_runoff_m equation will need to be adjusted by the fraction of pervious area.
        """
        a_Xinanjiang_inflection_point_parameter = 1
        b_Xinanjiang_shape_parameter = 1
        x_Xinanjiang_shape_parameter = 1

        if ((tension_water_m/max_tension_water_m) <= (0.5 - a_Xinanjiang_inflection_point_parameter)): 
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m * \
                (np.power((0.5 - a_Xinanjiang_inflection_point_parameter),\
                    (1.0 - b_Xinanjiang_shape_parameter)) * \
                        np.power((1.0 - (tension_water_m/max_tension_water_m)),\
                            b_Xinanjiang_shape_parameter))

        else: 
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m* \
                (1.0 - np.power((0.5 + a_Xinanjiang_inflection_point_parameter), \
                    (1.0 - b_Xinanjiang_shape_parameter)) * \
                        np.power((1.0 - (tension_water_m/max_tension_water_m)),\
                            (b_Xinanjiang_shape_parameter)))
    
        # Separate the surface water from the pervious runoff 
        ## NOTE: If impervious runoff is added to this subroutine, impervious runoff should be added to
        ## the surface_runoff_depth_m.
        
        cfe_state.surface_runoff_depth_m = pervious_runoff_m * \
             (1.0 - np.power((1.0 - (free_water_m/max_free_water_m)),x_Xinanjiang_shape_parameter))

        # The surface runoff depth is bounded by a minimum of 0 and a maximum of the water input depth.
        # Check that the estimated surface runoff is not less than 0.0 and if so, change the value to 0.0.
        if(cfe_state.surface_runoff_depth_m < 0.0): 
            cfe_state.surface_runoff_depth_m = 0.0;
    
        # Check that the estimated surface runoff does not exceed the amount of water input to the soil surface.  If it does,
        # change the surface water runoff value to the water input depth.
        if(cfe_state.surface_runoff_depth_m > cfe_state.timestep_rainfall_input_m): 
             cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
        
        # Separate the infiltration from the total water input depth to the soil surface.
        cfe_state.infiltration_depth_m = cfe_state.timestep_rainfall_input_m- cfe_state.surface_runoff_depth_m;    

        return
                            
    def et_from_soil(self,cfe_state):
        """
            Take AET from soil moisture storage, 
            using Budyko type curve to limit PET if wilting<soilmoist<field_capacity
        """
        if cfe_state.reduced_potential_et_m_per_timestep > 0:
            
            if cfe_state.soil_reservoir['storage_m'] >= cfe_state.soil_reservoir['storage_threshold_primary_m']:
            
                cfe_state.actual_et_from_soil_m_per_timestep = np.minimum(cfe_state.reduced_potential_et_m_per_timestep, 
                                                       cfe_state.soil_reservoir['storage_m'])
                               
            elif ((cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']) and 
                  (cfe_state.soil_reservoir['storage_m'] < cfe_state.soil_reservoir['storage_threshold_primary_m'])):
            
                Budyko_numerator = cfe_state.soil_reservoir['storage_m'] - cfe_state.soil_reservoir['wilting_point_m']
                Budyko_denominator = cfe_state.soil_reservoir['storage_threshold_primary_m'] - \
                                     cfe_state.soil_reservoir['wilting_point_m']
                Budyko = Budyko_numerator / Budyko_denominator

                cfe_state.actual_et_from_soil_m_per_timestep = np.minimum(Budyko * cfe_state.reduced_potential_et_m_per_timestep,cfe_state.soil_reservoir['storage_m'])
                               
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.reduced_potential_et_m_per_timestep -= cfe_state.actual_et_from_soil_m_per_timestep
        return
            
            
    def check_is_fabs_less_than_epsilon(self,cfe_state,epsilon=1.0e-9):
        """ in the instance of calling the gw reservoir the secondary flux should be zero- verify
            From Line 157 of https://github.com/NOAA-OWP/cfe/blob/master/original_author_code/cfe.c
        """
        a = cfe_state.secondary_flux
        if np.abs(a) < epsilon:
            cfe_state.is_fabs_less_than_epsilon = True
        else:
            print("problem with nonzero flux point 1\n")
            cfe_state.is_fabs_less_than_epsilon = False 
    
    def soil_moisture_flux_ode(self, t, S, cfe_state, reservoir):
        """
        Soil reservoir module that solves ODE
        Using ODE allows simultaneous calculation of outflux, instead of stepwise subtraction of flux which causes overextraction from SM reservoir
        The behavior of soil moisture storage is divided into 3 stages. 
        Stage 1: S (Soil moisture storage ) > storage_threshold_primary_m
            Interpretation: When the soil moisture is plenty, AET(=PET), percolation, and lateral flow are all active.
            Equation: dS/dt = Infiltration - PET - (Klf+Kperc) * (S - storage_threshold_primary_m)/(storage_max_m - storage_threshold_primary_m)
        Stage 2: storage_threshold_primary_m > S (Soil moisture storage) > storage_threshold_primary_m - wltsmc
            Interpretation: When the soil moisture is in the medium range, AET is active and proportional to the soil moisture storage ratio. No percolation and lateral flow fluxes. 
            Equation: dS/dt = Infiltration - PET * (S - wltsmc)/(storage_threshold_primary_m - wltsmc)
        Stage 3: wltsmc > S (Soil moisture storage)
            Interpretation: When the soil moisture is depleted, no outflux is active
            Equation: dS/dt = Infitlation
            
        :param t: time
        :param S: Soil moisture storage in meter
        :param storage_threshold_primary_m:
        :param storage_max_m: maximum soil moisture storage, i.e., porosity
        :param coeff_primary: K_perc, percolation coefficient
        :param coeff_secondary: K_lf, lateral flow coefficient
        :param PET: potential evapotranspiration
        :param infilt: infiltration
        :param wilting_point_m: wilting point (in meter)
        :return: dS
        """
        storage_above_threshold_m = S - reservoir['storage_threshold_primary_m']
        storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
        storage_ratio = np.minimum(storage_above_threshold_m / storage_diff, 1)

        perc_lat_switch = np.multiply(S - reservoir['storage_threshold_primary_m'] > 0, 1)
        ET_switch = np.multiply(S - reservoir['wilting_point_m'] > 0, 1)

        storage_above_threshold_m_paw = S - reservoir['wilting_point_m']
        storage_diff_paw = reservoir['storage_threshold_primary_m'] - reservoir['wilting_point_m']
        storage_ratio_paw = np.minimum(storage_above_threshold_m_paw/storage_diff_paw, 1) # Equation 11 (Ogden's document)
        dS = cfe_state.infiltration_depth_m -1 * perc_lat_switch * (reservoir['coeff_primary'] + reservoir['coeff_secondary']) * storage_ratio - ET_switch * cfe_state.reduced_potential_et_m_per_timestep * storage_ratio_paw
        return dS

    def jac(self, t, S, cfe_state, reservoir):
        # The Jacobian matrix of the equation conceptual_reservoir_flux_calc. Calculated as df/dS = (dS/dt)/dS.
        storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
    
        perc_lat_switch = np.multiply(S - reservoir['storage_threshold_primary_m'] > 0, 1)
        ET_switch = np.multiply((S - reservoir['wilting_point_m'] > 0) and (S - reservoir['storage_threshold_primary_m'] < 0), 1)
    
        storage_diff_paw = reservoir['storage_threshold_primary_m'] - reservoir['wilting_point_m']
    
        dfdS = -1 * perc_lat_switch * (reservoir['coeff_primary'] + reservoir['coeff_secondary']) * 1/storage_diff - ET_switch * cfe_state.reduced_potential_et_m_per_timestep * 1/storage_diff_paw
        return [dfdS]
    
    def soil_moisture_flux_calc_with_ode(self, cfe_state, reservoir):
        """
            This function solves the soil moisture mass balance.
            Inputs:
                reservoir
            Outputs:
                primary_flux_m (percolation)
                secondary_flux_m (lateral flow)
                actual_et_from_soil_m_per_timestep (et_from_soil)
        """

        # Initialization
        y0 = [reservoir['storage_m']]
        t = np.array([0, 0.05, 0.15, 0.3, 0.6, 1.0]) # ODE time descritization of one time step

        # Solve and ODE
        sol = odeint(
            self.soil_moisture_flux_ode,
            y0,
            t,
            args=(cfe_state, reservoir),
            tfirst=True,
            Dfun=self.jac
        )

        # Finalize results
        ts_concat = t
        ys_concat = np.concatenate(sol, axis=0)

        # Estimate fluxes at each ODE time descritization
        t_proportion = np.diff(ts_concat)
        ys_avg = np.convolve(ys_concat, np.ones(2), 'valid') / 2

        lateral_flux = np.zeros(ys_avg.shape)
        perc_lat_switch = ys_avg - reservoir['storage_threshold_primary_m'] > 0
        lateral_flux[perc_lat_switch] = reservoir['coeff_secondary'] * np.minimum(
            (ys_avg[perc_lat_switch] - reservoir['storage_threshold_primary_m']) / (
                        reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']), 1)
        lateral_flux_frac = lateral_flux * t_proportion

        perc_flux = np.zeros(ys_avg.shape)
        perc_flux[perc_lat_switch] = reservoir['coeff_primary'] * np.minimum(
            (ys_avg[perc_lat_switch] - reservoir['storage_threshold_primary_m']) / (
                        reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']), 1)
        perc_flux_frac = perc_flux * t_proportion

        et_from_soil = np.zeros(ys_avg.shape)
        ET_switch = ys_avg - cfe_state.soil_params['wltsmc']* cfe_state.soil_params['D'] > 0
        et_from_soil[ET_switch] = cfe_state.reduced_potential_et_m_per_timestep * np.minimum(
            (ys_avg[ET_switch] - cfe_state.soil_params['wltsmc']* cfe_state.soil_params['D']) / (reservoir['storage_threshold_primary_m'] - cfe_state.soil_params['wltsmc']* cfe_state.soil_params['D']), 1)
        et_from_soil_frac = et_from_soil * t_proportion

        infilt_to_soil = np.repeat(cfe_state.infiltration_depth_m, ys_avg.shape)
        infilt_to_soil_frac = infilt_to_soil * t_proportion

        # Scale fluxes (Since the sum of all the estimated flux above usually exceed the input flux because of calculation errors, scale it
        # The more finer ODE time descritization you use, the less errors you get, but the more calculation time it takes 
        
        # Get the scale factor
        sum_outflux = lateral_flux_frac + perc_flux_frac + et_from_soil_frac
        if sum_outflux.any() == 0:
            flux_scale = 0
            if cfe_state.infiltration_depth_m > 0:
                # To account for mass balance error by ODE
                final_storage_m = y0[0] + cfe_state.infiltration_depth_m
            else:
                final_storage_m = y0[0]
        else:
            flux_scale = (
                (ys_concat[0] - ys_concat[-1]) + np.sum(infilt_to_soil_frac)
            ) / np.sum(sum_outflux)
            final_storage_m = ys_concat[-1]

        # Scale the fluxes
        scaled_lateral_flux = lateral_flux_frac * flux_scale
        scaled_perc_flux = perc_flux_frac * flux_scale
        scaled_et_flux = et_from_soil_frac * flux_scale

        # Pass the results
        cfe_state.primary_flux_m = math.fsum(scaled_perc_flux)
        cfe_state.secondary_flux_m = math.fsum(scaled_lateral_flux)
        cfe_state.actual_et_from_soil_m_per_timestep = math.fsum(scaled_et_flux)
        # reservoir['storage_m'] = ys_concat[-1]
        cfe_state.soil_reservoir["storage_m"] = final_storage_m
        
        # # For debugging
        # print(f"cfe_state.infiltration_depth_m {cfe_state.infiltration_depth_m}")
        # print(f"cfe_state.primary_flux_m {cfe_state.primary_flux_m}")
        # print(f"cfe_state.secondary_flux_m {cfe_state.secondary_flux_m}")
        # print(f"cfe_state.actual_et_from_soil_m_per_timestep {cfe_state.actual_et_from_soil_m_per_timestep}")
        # print(f"cfe_state.soil_reservoir['storage_m'] {cfe_state.soil_reservoir['storage_m']}")
        
        return

        
    #    # Comment out because this section raises Runtime error, as dS_soil_reservoir is extremely small
    #    # dS based on Soil reservoir
    #    dS_soil_reservoir = ys_concat[-1]-ys_concat[0]
    #    # dS based on fluxes
    #    dS_fluxes = cfe_state.infiltration_depth_m - cfe_state.primary_flux_m - cfe_state.secondary_flux_m - cfe_state.actual_et_from_soil_m_per_timestep
    #    if ((dS_soil_reservoir - dS_fluxes) / dS_soil_reservoir) >= 0.01:
    #        warnings.warn(f'Mass balance error is more than 1%. \n dS({ys_concat[-1]-ys_concat[0]}) = I({cfe_state.infiltration_depth_m}) - Perc({cfe_state.primary_flux_m}) - Lat({cfe_state.secondary_flux_m}) - AET({cfe_state.actual_et_from_soil_m_per_timestep})')
        

    def step(self):
        cfe_state = self
        # Rainfall and ET 
        self.calculate_input_rainfall_and_PET(cfe_state)
        self.calculate_evaporation_from_rainfall(cfe_state)
        self.calculate_evaporation_from_soil(cfe_state)
        
        # Infiltration partitioning
        self.calculate_the_soil_moisture_deficit(cfe_state)
        self.calculate_infiltration_excess_overland_flow(cfe_state)
        self.calculate_saturation_excess_overland_flow_from_soil(cfe_state)
        self.track_infiltration_and_runoff(cfe_state)

        # Soil moisture reservoir
        self.run_soil_moisture_scheme(cfe_state)
        self.update_outflux_from_soil(cfe_state)
        
        # Groundwater reservoir
        self.calculate_groundwater_storage_deficit(cfe_state)
        self.calculate_saturation_excess_overland_flow_from_gw(cfe_state)
        self.track_volume_from_percolation_and_lateral_flow(cfe_state)
        self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.gw_reservoir)  
        self.set_flux_from_deep_gw_to_chan_m(cfe_state)
        self.check_is_fabs_less_than_epsilon(cfe_state) 
        self.remove_flux_from_deep_gw_to_chan_m(cfe_state)
        
        # Surface runoff rounting
        self.convolution_integral(cfe_state)
        self.track_volume_from_giuh(cfe_state)
        self.track_volume_from_deep_gw_to_chan(cfe_state)
        
        # Lateral flow rounting
        self.nash_cascade(cfe_state)
        self.track_volume_from_nash_cascade(cfe_state)
        self.add_up_total_flux_discharge(cfe_state)
        
        # Time
        self.update_current_time(cfe_state)