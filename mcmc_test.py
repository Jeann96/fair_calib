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
import sys
'''
fair_path = 'FAIR/src/'
if fair_path not in sys.path:
    sys.path.append(fair_path)
'''
cwd = os.getcwd()
from fair_tools import load_data, load_calib_samples, load_prior_samples, load_MC_samples, runFaIR, get_prior, read_temperature_data, get_param_ranges
from plotting_tools import plot_distributions, plot_temperature, plot_constraints
from mcmc_tools import mcmc_run, mcmc_extend
from netCDF4 import Dataset
#from dotenv import load_dotenv
#load_dotenv()

figdir = f"{cwd}/figures"
result_folder = 'MC_results'
#filename = 'optimal_sampling_long'
filename = 'very_long_sampling'
#filename = 'test'

# Choose FaIR scenario (do not change)
scenario = 'ssp245'
# Length of the warmup (burn-in) period
warmup = 1000
# Number of samples collected after the warmup
samples = 10000

# If true, removes the old file and starts samplin new one
new_chain = True
# If true, extends the existing chain equal to the number of {samples}
extend = False
# Covariance for the proposal distribution used for extending the chain
Ct = np.loadtxt('Ct')
# Include constraints for the sampling
use_constraints = True
# Plot figures
plot = True

### MAIN PROGRAM STARTS HERE
# Read prior and calibration configurations
calib_configs = load_calib_samples()
prior_configs = load_prior_samples()
# Parameter ranges
param_ranges = get_param_ranges()
# List of paramters which are not sampled using MCMC
excluded = ['ari CH4','ari N2O','ari NH3','ari NOx','ari VOC','ari Equivalent effective stratospheric chlorine','seed']
# Parameters which are sampled using MCMC
included = [param for param in prior_configs.columns if param not in excluded]
print(f'Parameters included for sampling:\n{included}')

if not os.path.exists(f'{cwd}/{result_folder}/{scenario}'):
    os.makedirs(f'{cwd}/{result_folder}/{scenario}')

# Read data
T_data, T_std = read_temperature_data()

solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(calib_configs),1750,2100)
prior_fun = get_prior(included)
fair_calib = runFaIR(solar_forcing,volcanic_forcing,emissions,calib_configs,scenario,
                     start=1750,end=2100)

model_var = 0.35**2
init_config = prior_configs.median(axis=0).to_frame().transpose()
seed = np.random.randint(*param_ranges['seed'])
init_config['seed'] = seed
C0 = np.diag(np.square(0.2*prior_configs[included].std(axis=0).to_numpy()))
#C0 = np.diag((0.01 * init_config[included].to_numpy().squeeze())**2)
solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2100)
fair_init = runFaIR(solar_forcing,volcanic_forcing,emissions,init_config,scenario,
                     start=1750,end=2100)
if new_chain:
    mcmc_run(scenario,init_config,included,samples,warmup,C0=C0,prior=prior_fun,use_constraints=use_constraints,
             model_var=model_var,filename=filename)
if extend:
    mcmc_extend(scenario,init_config,samples,prior=prior_fun,use_constraints=use_constraints, 
                model_var=model_var,Ct=Ct,filename=filename)

if plot:
    ds = Dataset(f'{result_folder}/{scenario}/{filename}.nc',mode='r')
    prior_loss, data_loss, loss = ds['prior_loss_chain'][:].data, ds['data_loss_chain'][:].data, ds['loss_chain'][:].data
    chain, seeds = ds['chain'][:,:].data, ds['seeds'][:].data
    
    MAP_index = warmup + np.argmin(loss[warmup:])
    MAP_config = init_config.copy()
    MAP_config[included] = chain[MAP_index,:]
    MAP_config['seed'] = seeds[MAP_index]
    
    full_pos_configs = load_MC_samples(ds,N=5000,thinning=1,param_ranges=param_ranges)
    #full_pos_configs.drop('CO2 concentration 1750',axis='columns',inplace=True)
    #full_pos_configs.rename({'co2_concentration_1750':'CO2 concentration 1750'},axis='columns',inplace=True)

    plot_distributions(prior_configs,calib_configs,alpha=0.6,title='Prior vs calibration',savefig=True)
    plot_distributions(prior_configs,full_pos_configs,alpha=0.6,title='Prior vs MC sampling',savefig=True)
    plot_distributions(calib_configs,full_pos_configs,alpha=0.6,title='Calib vs MC sampling',savefig=True)
    
    #Run fair with chosen number of configuration chosen from the parameter posterior
    pos_configs = load_MC_samples(ds,N=1000,thinning=5,param_ranges=param_ranges)
    #pos_configs.drop('CO2 concentration 1750',axis='columns',inplace=True)
    #pos_configs.rename({'co2_concentration_1750':'CO2 concentration 1750'},axis='columns',inplace=True)
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(pos_configs),1750,2100)
    fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,pos_configs,scenario,
                             start=1750,end=2100)
    
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(MAP_config),1750,2100)
    fair_MAP = runFaIR(solar_forcing,volcanic_forcing,emissions,MAP_config,scenario,
                       start=1750,end=2100)
    plot_temperature(scenario, 1750, 2100, fair_calib, fair_posterior, fair_MAP, T_data, T_std)
    plot_constraints(ds,N=1000,thinning=5)
    # Close dataset
    ds.close()
    #del sys.path[-1]