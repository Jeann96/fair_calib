#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:20:06 2024

@author: nurmelaj
"""

import os
import numpy as np
from data_handling import read_prior_samples,read_forcing
from fair_tools import setup_fair,run_configs,validate_config,temperature_anomaly,temperature_variation
from dotenv import load_dotenv
from xarray import DataArray
load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))

batch_size = 2000
seed = 1234

configs = read_prior_samples(dtype=np.float32)
configs['seed'] = 1234

# All possible FaIR scenarios
scenarios = ['ssp119','ssp126','ssp245','ssp370','ssp434','ssp460','ssp534-over','ssp585']

solar_forcing, volcanic_forcing = read_forcing(1750,2023,dtype=np.float32)
std_xr = DataArray(dims=['scenario','config'],
                   coords=dict(scenario=(['scenario'],scenarios),config=np.arange(samples,dtype=np.uint32)))

for n in range(samples // batch_size):
    print(f'Batch {n+1}/{samples // batch_size}')
    config_indices = np.arange(n*batch_size,(n+1)*batch_size,dtype=np.uint32)
    config_batch = configs.iloc[config_indices,:]
    valid = validate_config(config_batch)
    fair_allocated = setup_fair(scenarios,np.sum(valid),start=1750,end=2023,dtype=np.float32)
    fair_run = run_configs(fair_allocated,config_batch[valid],solar_forcing,volcanic_forcing,
                           start=1750,end=2023,stochastic_run=False)
    model_T = temperature_anomaly(fair_run,rolling_window_size=2)
    std_xr.loc[dict(config=config_indices[valid])] = temperature_variation(model_T).data
np.save(f"fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/T_variation_1750-2022.npy", 
        std_xr.mean(dim='scenario').data)
