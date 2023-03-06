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
#scenario = 'ssp245'
scenario = 'ssp119'
start = 1850
end = 2020

# MC parameters
warmup = 10000
samples = 100000
nconfigs = samples

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
T_prior = fair_prior.temperature.loc[dict(scenario=scenario,layer=0)].mean(axis=1)
prior_fun, prior_distributions = fit_priors(prior_configs,exclude=excluded)

# Mean vector of the prior config
mean_config = prior_configs.mean(axis=0).copy().to_frame().transpose()
x0 = mean_config[included].to_numpy().squeeze()
C0 = np.diag((0.1*prior_configs[included].std(axis=0).to_numpy())**2)
# Run mcmc, creates netcdf file
mcmc_run(scenario,x0,included,samples,warmup,C0=C0,prior=prior_fun,bounds=None)

#warmup = sampling['warmup']´

ds = Dataset(f'{result_folder}/{scenario}/sampling.nc',mode='r')
MAP, cov = ds['MAP'][:].data, ds['cov'][:,:].data
ds.close()
pos_configs = load_MC_configs('MC_results',scenario,prior_distributions,N=nconfigs)
plot_distributions(prior_distributions,prior_configs,pos_configs,exclude=excluded)

# Temperatur trend
solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2100)
opt_config = load_optimal_config('MC_results',scenario)
fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,opt_config,scenario,
                         start=1750,end=2100)
T_post = fair_posterior.temperature.loc[dict(scenario=scenario,layer=0)].mean(axis=1)
plot_temperature(scenario, start, end, T_prior, T_post, T_data, T_std)
#f.concentration.loc[dict(scenario=scenario, specie=specie)], label=f.configs)


