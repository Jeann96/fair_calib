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

fair_path = 'FAIR/src/'
if fair_path not in sys.path:
    sys.path.append(fair_path)

cwd = os.getcwd()
from fair_tools import runFaIR, get_param_ranges
# Import data reading tools
from fair_tools import read_forcing_data,read_calib_samples,read_prior_samples,read_MC_samples,read_gmst_temperature,read_hadcrut_temperature
from plotting_tools import plot_distributions, plot_temperature, plot_constraints
from mcmc_tools import mcmc_run, mcmc_extend
from netCDF4 import Dataset

figdir = f"{cwd}/figures"
result_folder = 'MC_results'
filename = 'detrend_chain'
#filename = 'stochastic_chain'
#filename = 'deterministic_chain'

# Choose FaIR scenario (do not change)
scenario = 'ssp245'
# Length of the warmup (burn-in) period
warmup = 1000
# Number of samples collected after the warmup
samples = 1000

# If true, removes the old file and starts samplin new one
new_chain = True
# If true, extends the existing chain equal to the number of {samples}
extend = False
# Covariance for the proposal distribution used for extending the chain
cov_file = 'Ct'
# Include prior and/or constraints for the sampling
use_prior = True
use_constraints = True
# Data loss method
data_loss_method = 'detrend'
# Stochastic or deterministic run
stochastic = True
# Plot figures and save figures
plot = False
save_figs = False

### MAIN PROGRAM STARTS HERE
# Read prior and calibration configurations
calib_configs = read_calib_samples()
prior_configs = read_prior_samples()
param_ranges = get_param_ranges()
param_means = dict(prior_configs.mean(axis='rows'))
param_stds = dict(prior_configs.std(axis='rows'))
# List of paramters which are not sampled using MCMC
excluded = ['gamma','sigma_xi','sigma_eta','ari CH4','ari N2O','ari NH3','ari NOx','ari VOC',
            'ari Equivalent effective stratospheric chlorine','seed']
#excluded = ['ari CH4','ari N2O','ari NH3','ari NOx','ari VOC',
#            'ari Equivalent effective stratospheric chlorine','seed']
# Parameters which are sampled using MCMC
included = [param for param in prior_configs.columns if param not in excluded]
print(f'Parameters included for sampling:\n{included}')
print(f'Filename: {filename}.nc')

if not os.path.exists(f'{cwd}/{result_folder}/{scenario}'):
    os.makedirs(f'{cwd}/{result_folder}/{scenario}')

# Read data
gmst = read_gmst_temperature()
T = read_hadcrut_temperature()

solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,len(calib_configs),1750,2100)
fair_calib = runFaIR(solar_forcing,volcanic_forcing,emissions,calib_configs,scenario,
                     start=1750,end=2100,stochastic_run=stochastic)

#model_var = 0.35**2
init_config = prior_configs.median(axis=0).to_frame().transpose()
seed = np.random.randint(*param_ranges['seed'])
init_config['seed'] = seed
#init_config['gamma'] = max(param_ranges['gamma'])

C0 = np.diag(np.square(0.2*prior_configs[included].std(axis=0).to_numpy()))
#C0 = np.diag((0.01 * init_config[included].to_numpy().squeeze())**2)
solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,1,1750,2100)
fair_init = runFaIR(solar_forcing,volcanic_forcing,emissions,init_config,scenario,
                     start=1750,end=2100,stochastic_run=stochastic)
if new_chain:
    mcmc_run(scenario,init_config,included,samples,warmup,C0=C0,use_constraints=use_constraints,use_prior=use_prior,
             stochastic_run=stochastic,filename=filename,data_loss_method=data_loss_method)
if extend:
    Ct = np.loadtxt(cov_file) if os.path.exists(cov_file) else None
    mcmc_extend(scenario,init_config,samples,use_constraints=use_constraints, 
                stochastic_run=stochastic,Ct=Ct,filename=filename,data_loss_method=data_loss_method)

if plot:
    ds = Dataset(f'{result_folder}/{scenario}/{filename}.nc',mode='r')
    loss = ds['loss_chain'][:].data
    chain, seeds = ds['chain'][:,:].data, ds['seeds'][:].data
    
    MAP_index = warmup + np.argmin(loss[warmup:])
    MAP_config = init_config.copy()
    MAP_config[included] = chain[MAP_index,:]
    MAP_config['seed'] = seeds[MAP_index]
    
    full_pos_configs = read_MC_samples(ds,param_ranges,param_means,param_stds,N=10000,thinning=50)
    plot_distributions(prior_configs,calib_configs,alpha=0.6,title='Prior vs calibration',savefig=save_figs)
    plot_distributions(prior_configs,full_pos_configs[included],alpha=0.6,title='Prior vs MC sampling',savefig=save_figs)
    plot_distributions(calib_configs,full_pos_configs[included],alpha=0.6,title='Calib vs MC sampling',savefig=save_figs)
    
    #Run fair with chosen number of configuration chosen from the parameter posterior
    pos_configs = read_MC_samples(ds,param_ranges,param_means,param_stds,N=2000,thinning=200)
    solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,len(pos_configs),1750,2100)
    fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,pos_configs,scenario,
                             start=1750,end=2100,stochastic_run=stochastic)
    
    solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,len(MAP_config),1750,2100)
    fair_MAP = runFaIR(solar_forcing,volcanic_forcing,emissions,MAP_config,scenario,
                       start=1750,end=2100,stochastic_run=stochastic)
    plot_temperature(scenario,1750,2100,fair_calib,posterior=fair_posterior,MAP=fair_MAP,
                     obs=gmst,obs_std=T.std(dim='realization'),savefig=save_figs)
    plot_constraints(ds,N=10000,thinning=50,savefig=save_figs)
    # Close dataset
    ds.close()
    del sys.path[-1]