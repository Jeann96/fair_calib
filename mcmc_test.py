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

from fair_tools import load_data, load_configs, load_MC_samples, runFaIR, get_prior, read_temperature_data, constraint_ranges, constraint_priors, constraint_targets, constraint_posteriors
from plotting_tools import plot_distributions, plot_temperature, plot_constraints
from mcmc_tools import mcmc_run, mcmc_extend
from netCDF4 import Dataset
from dotenv import load_dotenv
load_dotenv()
figdir = f"{cwd}/figures"
result_folder = 'MC_results'
filename = 'optimal_sampling'
#filename = 'test'

# Choose scenario
#scenario = 'ssp119'
#scenario = 'ssp126'
scenario = 'ssp245'
#start = 1850
#end = 2020

# MC parameters
warmup = 10000
samples = 200000
use_constraints = True
plot = True

prior_configs = load_configs()
# Parameter ranges as 99% confidence interval
param_ranges = {param: np.percentile(prior_configs[param],[0.5,99.5]) for param in prior_configs.columns}
# Adjust effective range of beta-parameter
param_ranges['beta'] = np.array([-2.1,-0.1])
#excluded = ['sigma_xi','sigma_eta','ari BC', 'ari CH4', 'ari N2O', 'ari NH3', 'ari NOx', 'ari OC', 
#            'ari Sulfur', 'ari VOC','ari Equivalent effective stratospheric chlorine','seed']
#excluded = ['sigma_eta','sigma_xi','seed']
excluded = ['ari CH4','ari N2O','ari NOx','ari VOC','ari Equivalent effective stratospheric chlorine','seed']
included = [param for param in prior_configs.columns if param not in excluded]


if not os.path.exists(f'{cwd}/{result_folder}/{scenario}'):
    os.makedirs(f'{cwd}/{result_folder}/{scenario}')

# Read data
T_data, T_std = read_temperature_data()

solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(prior_configs),1750,2100)
prior_fun = get_prior(included)
#prior_configs['sigma_xi'] = np.zeros(1001)
#prior_configs['sigma_eta'] = np.zeros(1001)
fair_prior = runFaIR(solar_forcing,volcanic_forcing,emissions,prior_configs,scenario,
                     start=1750,end=2100)
#model_var = fair_prior.temperature.loc[dict(timebounds=slice(1851,2021),scenario=scenario,layer=0)].std(dim='config').data**2
model_var = 0.2**2
init_config = prior_configs.median(axis=0).to_frame().transpose()
#init_config['sigma_eta'] = 0
#init_config['sigma_xi'] = 0
#seed = np.random.randint(prior_configs['seed'].min(),prior_configs['seed'].max()+1)
seed = np.random.randint(0,1e10+1)
init_config['seed'] = seed
#C0 = np.diag(prior_configs[included].std(axis=0).to_numpy()**2)
C0 = np.diag((0.1 * init_config[included].to_numpy().squeeze())**2)
solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2100)
fair_init = runFaIR(solar_forcing,volcanic_forcing,emissions,init_config,scenario,
                     start=1750,end=2100)

#mcmc_run(scenario,init_config,included,samples,warmup,C0=C0,prior=prior_fun,use_constraints=use_constraints,
#         model_var=model_var,filename=filename)
#mcmc_extend(scenario, 150000, prior=prior_fun, use_constraints=use_constraints, 
#            model_var=model_var, filename=filename)

if plot:
    ds = Dataset(f'{result_folder}/{scenario}/{filename}.nc',mode='r')
    
    prior_loss, data_loss, loss = ds['prior_loss_chain'][:].data, ds['data_loss_chain'][:].data, ds['loss_chain'][:].data
    chain, seeds = ds['chain'][:,:].data, ds['seeds'][:].data
    MAP_index = warmup + np.argmin(loss[warmup:])
    MAP_config = init_config.copy()
    MAP_config[included] = chain[MAP_index,:]
    MAP_config['seed'] = seeds[MAP_index]
    
    full_pos_configs = load_MC_samples(ds,N=50000)
    plot_distributions(prior_configs,full_pos_configs,exclude=excluded)
    
    #Run fair with chosen number of configuration chosen from the parameter posterior
    pos_configs = load_MC_samples(ds,N=1000)
    pos_configs[[name for name in excluded if name != 'seed']] = np.repeat(init_config[[name for name in excluded if name != 'seed']].to_numpy(),1000,axis=0)
    
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(pos_configs),1750,2100)
    fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,pos_configs,scenario,
                             start=1750,end=2100)
    #MAP_config['sigma_xi'] = np.median(prior_configs['sigma_xi'])
    #MAP_config['sigma_eta'] = np.median(prior_configs['sigma_eta'])
    #MAP_config['seed'] = np.random.randint(prior_configs['seed'].min(),prior_configs['seed'].max()+1)
    
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,len(MAP_config),1750,2100)
    fair_MAP = runFaIR(solar_forcing,volcanic_forcing,emissions,MAP_config,scenario,
                       start=1750,end=2100)
    plot_temperature(scenario, 1750, 2100, fair_prior, fair_posterior, fair_MAP, T_data, T_std)
    
    ranges = constraint_ranges()
    targets = constraint_targets()
    priors = constraint_priors()
    posteriors = constraint_posteriors()
    plot_constraints(ds,ranges,priors,targets,constraint_posteriors=posteriors)
    
    ds.close()