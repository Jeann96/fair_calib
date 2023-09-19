#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2023

@author: nurmelaj
"""

import numpy as np
from scipy.linalg import expm, eigh
import time
from fair_tools import load_data,runFaIR,read_temperature_data,constraint_targets,load_configs,compute_data_loss,compute_prior_loss,compute_constrained_loss,compute_constraints
from fair.energy_balance_model import EnergyBalanceModel
from netCDF4 import Dataset
import os
cwd = os.getcwd()

def validate_config(config,bounds=None):
    #These parameters require postivity and other constraints
    climate_response_params = ['gamma', 'c1', 'c2', 'c3', 'kappa1', 'kappa2', 'kappa3', 
                               'epsilon', 'sigma_eta', 'sigma_xi','F_4xCO2']
    # Climate response parameters must be positive
    positive = ((config[climate_response_params] >= 0).all(axis=1)[0])
    # Other conditions for paremeter to be physical
    physical = ((config['gamma'] > 0.8) & (config['c1'] > 2) & (config['c2'] > config['c1']) & (config['c3'] > config['c2']) & (config['kappa1'] > 0.3))[0]
    # The following covariance must be positive definite
    ebm = EnergyBalanceModel(ocean_heat_capacity=config.iloc[0, [1, 2, 3]],
                             ocean_heat_transfer=config.iloc[0, [4, 5, 6]],
                             deep_ocean_efficacy=config.iloc[0, 7],
                             gamma_autocorrelation=config.iloc[0, 0],
                             sigma_xi=config.iloc[0, 9],
                             sigma_eta=config.iloc[0, 8],
                             forcing_4co2=config.iloc[0, 10],
                             stochastic_run=True)
    eb_matrix = ebm._eb_matrix()
    q_mat = np.zeros((4, 4))
    q_mat[0, 0] = ebm.sigma_eta**2
    q_mat[1, 1] = (ebm.sigma_xi / ebm.ocean_heat_capacity[0]) ** 2
    h_mat = np.zeros((8, 8))
    h_mat[:4, :4] = -eb_matrix
    h_mat[:4, 4:] = q_mat
    h_mat[4:, 4:] = eb_matrix.T
    g_mat = expm(h_mat)
    q_mat_d = g_mat[4:, 4:].T @ g_mat[:4, 4:]
    positive_definite = np.all(eigh(q_mat_d,eigvals_only=True) > 0)
    # Bounds
    if bounds is not None:
        in_bounds = all((min(bounds[param]) <= config[param].iloc[0]) & (config[param].iloc[0] <= max(bounds[param])) for param in bounds.keys())
    else:
        in_bounds = True
    return positive & physical & positive_definite & in_bounds
    
def mcmc_run(scenario,init_config,names,samples,warmup,C0=None,prior=None,bounds=None,use_constraints=True,model_var=None,
             folder='MC_results',filename='sampling'):
    # Read data
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2101)
    T_data, T_std = read_temperature_data()
    targets = constraint_targets() 
    data, var = T_data.data, T_std.data**2
    if model_var is not None:
        var = var + model_var
    #ydim = len(data)
    # Run the first fair
    fair = runFaIR(solar_forcing,volcanic_forcing,emissions,init_config,scenario,
                   start=1750,end=2101)
    # Temperature anomaly compared to temperature mean between year 1850 and 1900
    anomaly = fair.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0).to_numpy().squeeze() \
            - fair.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
    # proposal fair configuration
    proposal_config = init_config.copy()
    progress, accepted = 0, 0
    x = init_config[names].to_numpy().squeeze()
    xdim = len(x)
    # Dimension related scaling for adaptive Metropolis
    sd = 2.4**2 / xdim
    if C0 is None:
        C0 = np.diag((0.1*x)**2)
    Ct = C0
    # MAP (maximum posterior) estimate, equal to minimum of the loss function
    MAP = x
    
    # Initialize chains
    chain = np.full((warmup+samples,xdim),np.nan)
    chain[0,:] = x
    data_loss_chain = np.full(warmup+samples,np.nan)
    prior_loss_chain = np.full(warmup+samples,np.nan)
    seed_chain = np.full(warmup+samples,np.nan)
    seed_chain[0] = int(init_config['seed'])
    constraints_chain = np.full((warmup+samples,9),np.nan)
    constraint_loss_chain = np.full((warmup+samples),np.nan)
    loss_chain = np.full(warmup+samples,np.nan)
    
    # Compute loss (negative log-likehood) from data, prior and constraints
    data_loss = compute_data_loss(anomaly, data, var)
    data_loss_chain[0] = data_loss
    prior_loss = compute_prior_loss(prior, x)
    prior_loss_chain[0] = prior_loss
    loss = data_loss + prior_loss
    loss_chain[0] = loss
    constraints = compute_constraints(fair)
    constraints_chain[0,:] = constraints
    if use_constraints:
        constraint_loss = compute_constrained_loss(constraints,targets)
        loss += constraint_loss
        
    # Create netcdf file for the results
    create_file(scenario,names,warmup,C0,filename=filename)
    
    start_time = time.process_time()
    t_start, t_end = 1, warmup+samples-1
    for t in range(t_start,t_end+1):
        if t == 1: print('Warmup period...')
        #proposal value for the chain
        proposal = np.random.multivariate_normal(x,Ct)
        proposal_config[names] = proposal
        seed = np.random.randint(0,1e10+1)
        proposal_config['seed'] = seed 
        valid = validate_config(proposal_config,bounds=bounds)
        if not valid:
            log_acceptance = -np.inf
        else:
            fair_proposal = runFaIR(solar_forcing,volcanic_forcing,emissions,proposal_config,scenario,
                                    start=1750,end=2101)
            anomaly = fair_proposal.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0).to_numpy().squeeze() \
                    - fair_proposal.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
            data_loss_proposal = compute_data_loss(anomaly, data, var)
            prior_loss_proposal = compute_prior_loss(prior, proposal)
            loss_proposal = data_loss_proposal + prior_loss_proposal
            proposal_constraints = compute_constraints(fair_proposal)
            if use_constraints:
                constraint_loss_proposal = compute_constrained_loss(proposal_constraints,targets)
                loss_proposal += constraint_loss_proposal
            log_acceptance = -loss_proposal + loss
        # Accept or reject
        #print(f'Acceptance: {min(100,100*np.exp(log_acceptance))} %, loss = {loss}')
        #print(log_acceptance)
        if np.log(np.random.uniform(0,1)) <= log_acceptance:
            x = proposal
            data_loss = data_loss_proposal
            prior_loss = prior_loss_proposal
            if loss_proposal < loss and t >= warmup:
                MAP = x
            loss = loss_proposal
            constraints = proposal_constraints
            if use_constraints:
                constraint_loss = constraint_loss_proposal
            accepted += 1
        chain[t,:] = x
        seed_chain[t] = seed
        data_loss_chain[t] = data_loss
        prior_loss_chain[t] = prior_loss
        constraints_chain[t,:] = constraints
        if use_constraints:
            constraint_loss_chain[t] = constraint_loss
        loss_chain[t] = loss
        
        # Adaptation starts after the warmup period
        if t == warmup - 1:
            print('...done')
            print(f'Acceptance ratio = {100*accepted/warmup:.2f} %')
            accepted = 0
            print('Sampling posterior...')
            Ct = sd * np.cov(chain[:warmup,:], rowvar=False)
            mean = chain[:warmup,:].mean(axis=0).reshape((xdim,1))
        elif t >= warmup:
            # Value in chain as column vector
            vec = chain[t,:].reshape((xdim,1))
            # Recursive update for mean
            next_mean = 1/(t+1) * (t*mean + vec)
            # Recursive update for the covariance matrix
            Ct = (t-1)/t * Ct + sd/t * (t * mean @ mean.T - (t+1) * next_mean @ next_mean.T + vec @ vec.T)
            # Update mean
            mean = next_mean
        # Print and save progress
        if (t-warmup+1) / samples * 100 >= progress:
            print(f'{progress}%')
            # Save results after each 10 %
            save_progress(scenario,warmup,Ct,chain[:(t+1),:],data_loss_chain[:(t+1)],prior_loss_chain[:(t+1)],
                          loss_chain[:(t+1)],seed_chain[:(t+1)],MAP,
                          constraints_arr = constraints_chain[:(t+1),:],filename=filename)
            progress += 10
    print('...done')
    end_time = time.process_time()
    print(f"AM performance:\nposterior samples = {samples}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")
    
def mcmc_extend(scenario, samples, prior=None, bounds=None, use_constraints=True, model_var=None,
                folder='MC_results', filename='sampling'):
    '''
    Extends the existing chains with provided number of samples.
    '''
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2101)
    data, std = read_temperature_data()
    targets = constraint_targets() 
    # Data variance
    var = std.to_numpy()**2
    if model_var is not None:
        var = var + model_var
    ds = Dataset(f'{cwd}/{folder}/{scenario}/{filename}.nc','r')
    params = ds['param'][:].tolist()
    xdim = ds.dimensions['param'].size
    sd = 2.4**2 / xdim
    chain, loss_chain = ds['chain'][:,:].data, ds['loss_chain'][:].data
    mean = chain.mean(axis=0).reshape((xdim,1))
    constraints_chain = np.column_stack(tuple(ds[constraint][:] for constraint in 
                                              ['ecs','tcr','T 1995-2014','ari','aci','aer', 'CO2', 'ohc', 'T 2081-2100']))
    data_loss_chain = ds['data_loss_chain'][:].data
    prior_loss_chain = ds['prior_loss_chain'][:].data
    constraint_loss_chain = ds['constraint_loss_chain'][:].data
    MAP = ds['MAP'][:].data
    Ct = ds['Ct'][:,:].data
    warmup = int(ds['warmup'][:].data)
    seed_chain = ds['seeds'][:].data
    t_start, t_end = ds.dimensions['sample'].size, ds.dimensions['sample'].size + samples - 1
    ds.close()
    
    x = chain[-1,:]
    constraints = constraints_chain[-1,:]
    prior_configs = load_configs()
    config = prior_configs.median(axis=0).to_frame().transpose()
    config['seed'] = seed_chain[-1]
    config[params] = x 
    #config = pd.DataFrame(data=x.tolist()+[seed_chain[-1]],index=params+['seed']).transpose()
    loss = loss_chain[-1]
    prior_loss = prior_loss_chain[-1]
    data_loss = data_loss_chain[-1]
    constraint_loss = constraint_loss_chain[-1]
    fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                   start=1750,end=2101)
    # Temperature anomaly compared to temperature mean between year 1850 and 1900
    anomaly = fair.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0).to_numpy().squeeze() \
            - fair.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
    MAP_index = np.argmin(loss_chain)
    loss_MAP = loss_chain[MAP_index]
    MAP = chain[MAP_index,:]
    MAP_config = config.copy()
    MAP_config[params] = MAP

    # Extend chains    
    chain = np.append(chain, np.full((samples,xdim),np.nan), axis=0)
    constraints_chain = np.append(constraints_chain,np.full((samples,9),np.nan), axis=0)
    seed_chain = np.append(seed_chain, np.full(samples,np.nan), axis=0)
    loss_chain = np.append(loss_chain, np.full(samples,np.nan), axis=0)
    data_loss_chain = np.append(data_loss_chain, np.full(samples,np.nan), axis=0)
    prior_loss_chain = np.append(prior_loss_chain, np.full(samples,np.nan), axis=0)
    constraint_loss_chain = np.append(constraint_loss_chain, np.full(samples,np.nan), axis=0)
    
    proposal_config = MAP_config.copy()
    progress, accepted = 0, 0
    start_time = time.process_time()
    print('Sampling posterior...')
    for t in range(t_start,t_end+1):
        #proposal value for the chain
        proposal = np.random.multivariate_normal(x,Ct)
        proposal_config[params] = proposal
        seed = np.random.randint(0,1e10+1)
        proposal_config['seed'] = seed
        valid = validate_config(proposal_config,bounds=bounds)
        prior_value = prior(proposal)
        if not valid or prior_value == 0.0:
            log_acceptance = -np.inf
        else:
            fair_proposal = runFaIR(solar_forcing,volcanic_forcing,emissions,proposal_config,scenario,
                                    start=1750,end=2101)
            anomaly = fair_proposal.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0).to_numpy().squeeze() \
                    - fair_proposal.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
            data_loss_proposal = compute_data_loss(anomaly, data, var)
            prior_loss_proposal = compute_prior_loss(prior, proposal)
            loss_proposal = data_loss_proposal + prior_loss_proposal
            proposal_constraints = compute_constraints(fair_proposal)
            if use_constraints:
                constraint_loss_proposal = compute_constrained_loss(proposal_constraints,targets)
                loss_proposal += constraint_loss_proposal
            log_acceptance = -loss_proposal + loss
        #Accept or reject
        if np.log(np.random.uniform(0,1)) <= log_acceptance:
            x = proposal
            if loss_proposal < loss_MAP:
                MAP = x
                loss_MAP = loss_proposal
            loss = loss_proposal
            prior_loss = prior_loss_proposal
            data_loss = data_loss_proposal
            constraints = proposal_constraints
            if use_constraints:
                constraint_loss = constraint_loss_proposal
            accepted += 1
        # Add new values to chains
        chain[t,:] = x
        seed_chain[t] = seed
        data_loss_chain[t] = data_loss
        prior_loss_chain[t] = prior_loss
        loss_chain[t] = loss
        constraints_chain[t,:] = constraints
        if use_constraints:
            constraint_loss_chain[t] = constraint_loss
        
        #Value in chain as column vector
        vec = chain[t,:].reshape((xdim,1))
        #Recursive update for mean
        next_mean = 1/(t+1) * (t*mean + vec)
        # Recursive update for the covariance matrix
        Ct = (t-1)/t * Ct + sd/t * (t * mean @ mean.T - (t+1) * next_mean @ next_mean.T + vec @ vec.T)
        #Update mean
        mean = next_mean
        # Print and save progress
        if (t-t_start+1) / samples * 100 >= progress:
            print(f'{progress}%')
            # Save results after each 10 %
            save_progress(scenario,warmup,Ct,chain[:(t+1),:],data_loss_chain[:(t+1)],prior_loss_chain[:(t+1)],
                          loss_chain[:(t+1)],seed_chain[:(t+1)],MAP,
                          constraints_arr = constraints_chain[:(t+1),:],filename=filename)
            progress += 10
    print('...done')
    end_time = time.process_time() 
    print(f"AM performance:\nposterior samples = {t_end+1-warmup}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")

def create_file(scenario,params,warmup,C0,filename='sampling'):
    N = len(params)
    ncfile = Dataset(f'MC_results/{scenario}/{filename}.nc',mode='w')
    ncfile.createDimension('sample', None)
    ncfile.createDimension('param', N)
    ncfile.title = 'FaIR MC sampling'
    ncfile.createVariable('sample', int, ('sample',),fill_value=False)
    ncfile.createVariable('param', str, ('param',),fill_value=False)
    ncfile['param'][:] = np.array(params,dtype=str)
    ncfile.createVariable('chain',float,('sample','param'),fill_value=False)
    ncfile.createVariable('data_loss_chain',float,('sample',),fill_value=False)
    ncfile.createVariable('prior_loss_chain',float,('sample',),fill_value=False)
    ncfile.createVariable('loss_chain',float,('sample',),fill_value=False)
    ncfile.createVariable('seeds',int,('sample',),fill_value=False)
    ncfile.createVariable('MAP',float,('param',),fill_value=False)
    ncfile.createVariable('cov',float,('param','param'),fill_value=False)
    ncfile.createVariable('Ct',float,('param','param'),fill_value=False)
    ncfile['Ct'][:,:] = C0
    ncfile.createVariable('warmup',int,fill_value=False)
    ncfile['warmup'][:] = warmup
    for constraint in ['ecs','tcr','T 1995-2014','ari','aci','aer','CO2','ohc','T 2081-2100']:
        ncfile.createVariable(constraint,float,('sample',),fill_value=np.nan)
    ncfile.createVariable('constraint_loss_chain',float,('sample',),fill_value=0.0)
    ncfile.close()

def save_progress(scenario,warmup,Ct,chain,data_loss_chain,prior_loss_chain,loss_chain,seeds,MAP,
                  constraints_arr=None,folder='MC_results',filename='sampling'):
    N = len(chain)
    ncfile = Dataset(f'{folder}/{scenario}/{filename}.nc',mode='a')
    index = ncfile['sample'][:]
    if len(index) == 0:
        ncfile['sample'][:] = np.arange(0,N,1,dtype=int)
        ncfile['chain'][0:,:] = chain
        ncfile['data_loss_chain'][0:N] = data_loss_chain
        ncfile['prior_loss_chain'][0:N] = prior_loss_chain
        ncfile['loss_chain'][0:N] = loss_chain
        ncfile['seeds'][0:N] = seeds.astype(int)
        for i, constraint in enumerate(['ecs','tcr','ohc','T 1995-2014','ari','aci','aer', 'CO2', 'T 2081-2100']):
            ncfile[constraint][0:N] = constraints_arr[:,i]
        ncfile['constraint_loss_chain'][0:N] = loss_chain - (prior_loss_chain + data_loss_chain)
    else:
        ncfile['sample'][:] = np.append(index,np.arange(index[-1]+1,N,1,dtype=int))
        ncfile['chain'][(index[-1]+1):N,:] = chain[(index[-1]+1):,:]
        ncfile['data_loss_chain'][(index[-1]+1):N] = data_loss_chain[(index[-1]+1):]
        ncfile['prior_loss_chain'][(index[-1]+1):N] = prior_loss_chain[(index[-1]+1):]
        ncfile['loss_chain'][(index[-1]+1):N] = loss_chain[(index[-1]+1):]
        ncfile['seeds'][(index[-1]+1):N] = seeds[(index[-1]+1):].astype(int)
        for i, constraint in enumerate(['ecs','tcr','T 1995-2014','ari','aci','aer', 'CO2', 'ohc', 'T 2081-2100']):
            ncfile[constraint][(index[-1]+1):N] = constraints_arr[(index[-1]+1):,i]
        ncfile['constraint_loss_chain'][(index[-1]+1):N] = loss_chain[(index[-1]+1):] - (prior_loss_chain[(index[-1]+1):] + data_loss_chain[(index[-1]+1):])
    ncfile['Ct'][:,:] = Ct
    ncfile['MAP'][:] = MAP
    ncfile.close()