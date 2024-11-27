#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 7 14:39:15 2023
​
@authors: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
          Janne Nurmela (janne.nurmela@fmi.fi)
"""

import os
cwd = os.getcwd()
import numpy as np
from fair_tools import setup_fair,run_configs,validate_config,resample_configs
from plotting_tools import plot_distributions,plot_temperature,plot_constraints
from mcmc_tools import mcmc_run
from data_handling import read_settings,read_calib_samples,read_prior_samples,transform_df,read_temperature_obs,read_forcing,make_plots

#filename = 'deterministic_chain_wss'
#filename = 'constraints_deterministic_chain'
#filename = 'deterministic_chain_fullyconstrained'
#filename = 'testrun'

#settings_file = 'server_settings.txt'
#settings_file = 'settings_deterministic_wss.txt'
#settings_file = 'settings_constraints_deterministic_wss.txt'
#settings_file = 'settings_deterministic_fullyconstrained.txt'
settings_file = 'settings.txt'
new_chain = True

if settings_file is not None:
    settings = read_settings(settings_file)
    filename = settings['filename']
    scenario = settings['scenario']
    warmup, nsamples = settings['warmup'], settings['samples']
    thinning = settings['thinning']
    parallel_chains = settings['parallel_chains']
    use_priors = settings['use_priors']
    use_constraints = settings['use_constraints']
    use_Tvar = settings['use_Tvar_constraint']
    stochastic = settings['stochastic']
else:
    filename = 'sampling'
    scenario = 'ssp245'
    warmup, nsamples = 10000, 100000
    use_priors = True
    use_constraints = True
    use_Tvar = True
    stochastic = False

### MAIN PROGRAM STARTS HERE
# Read prior and calibration configurations
prior_configs = read_prior_samples(validated=True)
CS_calib_configs = read_calib_samples()

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


prior_configs_transformed = transform_df(prior_configs,log_scale_transformation,name_changes=name_changes)
calib_configs_transformed = transform_df(CS_calib_configs,log_scale_transformation,name_changes=name_changes).dropna(axis='index')

# List of paramters which are not sampled using MCMC
excluded = ['seed']
# Parameters which are sampled using MCMC
included = [param for param in prior_configs_transformed.columns if param not in excluded]

# Read data
obs, unc = read_temperature_obs()

solar_forcing, volcanic_forcing = read_forcing(1750,2100)
N = 2000
# Resample prior and calib distributions
prior_configs_resampled = resample_configs(prior_configs_transformed,N=5*N)
prior_configs_resampled['seed'] = 12345
valids = validate_config(prior_configs_resampled,stochastic_run=True)
prior_configs_resampled = prior_configs_resampled.iloc[np.random.choice(np.where(valids)[0],size=N,replace=False),:]
fair_prior = run_configs(setup_fair([scenario],N,start=1750,end=2100),
                         transform_df(prior_configs_resampled,exp_scale_transformation,
                                      name_changes = {'log(-aci_beta)': 'aci_beta',
                                                      'log(aci_shape_SO2)': 'aci_shape_SO2',
                                                      'log(aci_shape_BC)': 'aci_shape_BC',
                                                      'log(aci_shape_OC)': 'aci_shape_OC'}), 
                         solar_forcing, volcanic_forcing, start=1750, end=2100)
#calib_configs_resampled = resample_configs(calib_configs_transformed,N=5*N)
valids = validate_config(calib_configs_transformed,stochastic_run=True)
calib_configs = calib_configs_transformed.iloc[valids,:].reset_index(drop=True)

fair_calib = run_configs(setup_fair([scenario],len(calib_configs),start=1750,end=2100),
                         transform_df(calib_configs,exp_scale_transformation,
                                      name_changes = {'log(-aci_beta)': 'aci_beta',
                                                      'log(aci_shape_SO2)': 'aci_shape_SO2',
                                                      'log(aci_shape_BC)': 'aci_shape_BC',
                                                      'log(aci_shape_OC)': 'aci_shape_OC'}), 
                         solar_forcing, volcanic_forcing, start=1750, end=2100)
#plot_temperature(fair_prior,fair_other_run=fair_calib,start=1750,end=2100,obs=obs,labels=['prior','calib','gmst'],savefig=False)

init_config = prior_configs_transformed.median(axis='rows').to_frame().transpose()
init_config['seed'] = 123456

if new_chain:
    print(f'Starting new chain with filename "{scenario}_{filename}.nc"')
    mcmc_run(scenario,init_config,included,nsamples,warmup,use_priors=use_priors,use_constraints=use_constraints,use_Tvar=use_Tvar,
             stochastic_run=stochastic,parallel_chains=parallel_chains,thinning=thinning,filename=f'{scenario}_{filename}')
    
#pos_configs = read_sampling_configs(f'{samplingdir}/{scenario}_{filename}.nc',N=N)
'''
fair_pos = run_configs(setup_fair([scenario],len(pos_configs),start=1750,end=2100),
                       transform_df(pos_configs,exp_scale_transformation,
                                    name_changes = {'log(-aci_beta)': 'aci_beta',
                                                    'log(aci_shape_SO2)': 'aci_shape_SO2',
                                                    'log(aci_shape_BC)': 'aci_shape_BC',
                                                    'log(aci_shape_OC)': 'aci_shape_OC'}), 
                         solar_forcing, volcanic_forcing, start=1750, end=2100, stochastic_run=stochastic)
'''

'''
if make_plots:    
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