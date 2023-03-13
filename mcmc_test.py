#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 7 14:39:15 2023
​
@authors: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
          Janne Nurmela (janne.nurmela@fmi.fi)
"""

import numpy as np
import os
cwd = os.getcwd()

from fair_tools import load_data, load_configs, load_MC_configs, load_optimal_config, runFaIR, fit_priors, read_temperature_data
from plotting_tools import plot_distributions, plot_temperature
from mcmc_tools import mcmc_run
from netCDF4 import Dataset

from dotenv import load_dotenv
load_dotenv()
figdir=f"{cwd}/figures"
result_folder = 'MC_results'


#fair_calibration_dir=f"{cwd}/fair-calibrate"

# Choose scenario

#scenario = 'ssp119'
scenario = 'ssp126'
#scenario = 'ssp245'
start = 1850
end = 2020

# MC parameters
warmup = 1000
samples = 100

prior_configs = load_configs()
excluded = ['sigma_xi','sigma_eta','ari BC', 'ari CH4', 'ari N2O', 'ari NH3', 'ari NOx', 'ari OC', 'ari Sulfur', 'ari VOC',
            'ari Equivalent effective stratospheric chlorine','seed']
included = [param for param in prior_configs.columns if param not in excluded]

if not os.path.exists(f'{cwd}/{result_folder}/{scenario}'):
    os.makedirs(f'{cwd}/{result_folder}/{scenario}')

solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(prior_configs),1750,2100)
T_data, T_std = read_temperature_data()
fair_prior = runFaIR(solar_forcing,volcanic_forcing,emissions,prior_configs,scenario,
                     start=1750,end=2100)
prior_fun, prior_distributions = fit_priors(prior_configs,exclude=excluded)

# Mean vector of the prior config

init_config = prior_configs.median(axis=0).to_frame().transpose()
for param in excluded:
    if param in ['sigma_eta','sigma_xi']:
        init_config[param] = 0.0
    else:
        init_config[param] = np.random.uniform(prior_configs[param].min(),prior_configs[param].max())
C0 = np.diag(prior_configs[included].std(axis=0).to_numpy()**2)
mcmc_run(scenario,init_config,included,samples,warmup,C0=C0,prior=prior_fun)
#mcmc_extend(scenario, 100, MAP_config, prior=prior_fun, folder='MC_results')

ds = Dataset(f'{result_folder}/{scenario}/sampling.nc',mode='r')
MAP, cov = ds['MAP'][:].data, ds['cov'][:,:].data
chain, wss, loss = ds['chain'][:,:].data, ds['wss_chain'][:].data, ds['loss_chain'][:].data
ds.close()
full_pos_configs = load_MC_configs('MC_results',scenario,included,prior_distributions)
plot_distributions(prior_distributions,prior_configs,full_pos_configs,exclude=excluded)

#Run fair with chosen number of configuration chosen from the parameter posterior
nconfigs = samples
solar_forcing, volcanic_forcing, emissions = load_data(scenario,nconfigs,1750,2100)
pos_configs = load_MC_configs('MC_results',scenario,included,prior_distributions,N=nconfigs)
fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,pos_configs,scenario,
                         start=1750,end=2100)

MAP_config = load_optimal_config('MC_results',scenario)
solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(MAP_config),1750,2100)
fair_opt = runFaIR(solar_forcing,volcanic_forcing,emissions,MAP_config,scenario,
                   start=1750,end=2100)
plot_temperature(scenario, start, end, fair_prior, fair_posterior, fair_opt, T_data, T_std)


