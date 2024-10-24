#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 7 14:39:15 2023
​
@authors: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
          Janne Nurmela (janne.nurmela@fmi.fi)
"""

import sys
import os
#import warnings
#warnings.simplefilter("ignore",ResourceWarning)
import numpy as np
from pandas import concat
from fair_tools import setup_fair,run_configs,get_param_ranges,temperature_variation
from plotting_tools import plot_distributions,plot_temperature,plot_constraints
from mcmc_tools import mcmc_run
from data_handling import cwd,transform_df
from data_handling import read_settings,read_forcing,read_calib_samples,read_prior_samples,read_temperature_obs
figdir = f"{cwd}/figures"
result_folder = 'MC_results'
filename = 'test'
#filename = 'stochastic_chain'
#filename = 'deterministic_chain_wss'
#filename = 'server_chain'

settings_file = 'settings_deterministic_wss.txt'
use_settings_file = True

if use_settings_file:
    settings = read_settings(settings_file)
    warmup, nsamples = settings['warmup'], settings['samples']
    thinning = settings['thinning']
    new_chain = True
    extend = False
    use_default_priors = settings['use_default_priors']
    use_default_constraints = settings['use_default_constraints']
    use_T_variability = settings['T_variability_constraint']
    target_log_likelihood = settings['target_log_likelihood']
    stochastic = settings['stochastic']
    plot = settings['plot']
    scenario = settings['scenario']
else:
    scenario = 'ssp245'
    warmup, nsamples = 100000, 1000000
    new_chain = False
    extend = False
    cov_file = 'Ct'
    use_default_priors = True
    use_default_constraints = True
    T_variability_constraint = False
    target_log_likelihood = 'wss'
    stochastic = False
    plot = True

### MAIN PROGRAM STARTS HERE
# Read prior and calibration configurations
#scenarios = ['ssp126','ssp245','ssp370']
#scenarios = ['ssp126']

prior_configs_orig = read_prior_samples(dtype=np.float64)
calib_configs_orig = read_calib_samples(dtype=np.float64)

log_scale_transformation = {'aci_beta': lambda x: np.log(-x),
                            'aci_shape_SO2': lambda x: np.log(x),
                            'aci_shape_BC': lambda x: np.log(x),
                            'aci_shape_OC': lambda x: np.log(x)}
name_changes = {'aci_beta': 'log(-aci_beta)',
                'aci_shape_SO2': 'log(aci_shape_SO2)',
                'aci_shape_BC': 'log(aci_shape_BC)',
                'aci_shape_OC': 'log(aci_shape_OC)'}
exp_scale_transformation = {'log(-aci_beta)': lambda x: -np.exp(x),
                            'log(aci_shape_SO2)': lambda x: np.exp(x),
                            'log(aci_shape_BC)': lambda x: np.exp(x),
                            'log(aci_shape_OC)': lambda x: np.exp(x)}


prior_configs_transformed = transform_df(prior_configs_orig,log_scale_transformation,name_changes=name_changes)
calib_configs_transformed = transform_df(calib_configs_orig,log_scale_transformation,name_changes=name_changes).dropna(axis='index')

param_ranges = get_param_ranges()
#param_means = dict(prior_configs.mean(axis='rows'))
#param_stds = dict(prior_configs.std(axis='rows'))

# List of paramters which are not sampled using MCMC
excluded = ['seed']
# Parameters which are sampled using MCMC
included = [param for param in prior_configs_transformed.columns if param not in excluded]
#print(f'Parameters included for sampling:\n{included}')
os.makedirs(f'{cwd}/{result_folder}', exist_ok=True)
    
C0 = np.diag(np.square(prior_configs_transformed[included].std(axis='rows').to_numpy() / 1.96))

# Read data
#obs, unc = read_temperature_obs()
#gmst = read_gmst_temperature()
#T, T_unc = read_hadcrut()
#solar_forcing, volcanic_forcing = read_forcing(1750,2100)

#fair_setup = init_fair([scenario],len(calib_configs_orig),emissions,start=1750,end=2100)
#fair_calib = run_configs(fair_setup,calib_configs_orig,solar_forcing,volcanic_forcing,start=1750,end=2100)
#plot_temperature(fair_calib,start=1750,end=2100,obs=T,obs_std=T_unc,savefig=False)

init_config = prior_configs_transformed.median(axis='rows').to_frame().transpose()
init_config['seed'] = 12345
#seed = np.random.randint(min(param_ranges['seed']),max(param_ranges['seed']))


'''
param = 'clim_sigma_xi'
configs = concat([init_config]*101, ignore_index=True, axis='rows')
configs[param] = np.linspace(min(param_ranges[param]),max(param_ranges[param]),101)
'''
'''
fair_allocated = setup_fair([scenario], 1, start=1750, end=2100)
fair_init = run_configs(fair_allocated,transform_df(init_config,exp_scale_transformation,
                                                    name_changes={'log(-aci_beta)':'aci_beta',
                                                                  'log(aci_shape_SO2)':'aci_shape_SO2',
                                                                  'log(aci_shape_BC)':'aci_shape_BC',
                                                                  'log(aci_shape_OC)':'aci_shape_OC'}),
                        solar_forcing,volcanic_forcing,start=1750,end=2100)
'''

#fair_init = run_configs(fair_setup,init_config,solar_forcing,volcanic_forcing,start=1750,end=2100)
sys.exit()
if new_chain:
    print(f'Starting new chain with filename "{filename}.nc"')
    mcmc_run(scenario,init_config,included,nsamples,warmup,thinning=thinning,C0=C0,default_constraints=use_default_constraints,
             default_priors=use_default_priors,T_variability=use_T_variability,
             stochastic_run=stochastic,target_log_likelihood=target_log_likelihood,filename=filename)
'''
if extend:
    print(f'Appending to existing chain {filename}.nc')
    Ct = np.loadtxt(cov_file) if os.path.exists(cov_file) else None
    mcmc_extend(scenario,init_config,samples,use_constraints=use_constraints, 
                stochastic_run=stochastic,Ct=Ct,filename=filename,data_loss_method=data_loss_method)
'''

ds = read_chain(f'{result_folder}/{scenario}/{filename}.nc')
loss = ds['loss_chain'][:].data
chain, seeds = ds['chain'][:,:].data, ds['seeds'][:].data


'''
if plot:    
    MAP_index = warmup + np.argmin(loss[warmup:])
    MAP_config = init_config.copy()
    MAP_config[included] = chain[MAP_index,:]
    MAP_config['seed'] = seeds[MAP_index]
    
    full_pos_configs = read_MC_samples(ds,param_ranges,param_means,param_stds,N=2000,thinning=1)
    plot_distributions(prior_configs,calib_configs,alpha=0.6,title='Prior vs Calib',savefig=True)
    plot_distributions(prior_configs,full_pos_configs[included],alpha=0.6,title='Prior vs MC sampling',savefig=True)
    plot_distributions(calib_configs,full_pos_configs[included],alpha=0.6,title='Calib vs MC sampling',savefig=True)
    
    #Run fair with chosen number of configuration chosen from the parameter posterior
    pos_configs = read_MC_samples(ds,param_ranges,param_means,param_stds,N=2000,thinning=1)
    solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,len(pos_configs),1750,2100)
    fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,pos_configs,scenario,
                             start=1750,end=2100,stochastic_run=True)
    solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,len(MAP_config),1750,2100)
    fair_MAP = runFaIR(solar_forcing,volcanic_forcing,emissions,MAP_config,scenario,
                       start=1750,end=2100,stochastic_run=True)
    plot_temperature(fair_calib,fair_other_run=None,start=1750,end=2100,MAP=None,
                     obs=gmst,obs_std=T.std(dim='realization'),savefig=False)
    plot_constraints(ds,N=2000,thinning=1,savefig=True)
# Close dataset
ds.close()
'''