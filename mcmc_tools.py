#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2023

@author: nurmelaj
"""

import numpy as np
import numpy.random as npr
import time
# Functions realted to the observations from fair_tools
from fair_tools import compute_trends
# Sampling and computation related functions from fair_tools
from fair_tools import setup_fair,run_configs,compute_data_loss,compute_constraints,temperature_anomaly,validate_config
from fair_tools import get_log_constraint_target,get_log_prior
from data_handling import read_forcing,read_temperature_obs,transform_df,create_file,save_progress
from xarray import DataArray
from pandas import concat
    
def mcmc_run(scenario,init_config,included,nsamples,warmup,C0=None,bounds=None,default_constraints=True,default_priors=True,
             T_variability=True,stochastic_run=False,target_log_likelihood='wss',filename='sampling',parallel_chains=10,thinning=10):
    # Read data
    solar_forcing, volcanic_forcing = read_forcing(1750,2023)
    
    if target_log_likelihood == 'wss':
        obs, unc = read_temperature_obs()
        var = unc**2
    elif target_log_likelihood == 'trend_line':
        trends = compute_trends()
        obs, var = np.mean(trends), np.std(trends)**2
    else:
        raise ValueError(f'Unknown target log likelihood {target_log_likelihood}')
    transformation = {'log(-aci_beta)': lambda x: -np.exp(x),
                      'log(aci_shape_SO2)': lambda x: np.exp(x),
                      'log(aci_shape_BC)': lambda x: np.exp(x),
                      'log(aci_shape_OC)': lambda x: np.exp(x)}
    name_changes={'log(-aci_beta)':'aci_beta',
                  'log(aci_shape_SO2)':'aci_shape_SO2',
                  'log(aci_shape_BC)':'aci_shape_BC',
                  'log(aci_shape_OC)':'aci_shape_OC'}
    # Allocate initial fair which can be used to run different configs
    fair_allocated = setup_fair([scenario],parallel_chains,start=1750,end=2023)
    configs = concat([init_config] * parallel_chains, ignore_index=True)
    fair_run = run_configs(fair_allocated,transform_df(configs,transformation,name_changes=name_changes),
                           solar_forcing,volcanic_forcing,start=1750,end=2023,stochastic_run=False)
    # Temperature anomaly compared to temperature mean between years 1850 and 1900
    model_T = temperature_anomaly(fair_run,rolling_window_size=2).squeeze(dim='scenario',drop=True)
    progress = 0
    accepted = np.zeros(parallel_chains,dtype=np.uint32)
    xdim = len(included)
    xt = configs[included].to_numpy()
    # Dimension related scaling for adaptive Metropolis algorithm
    sd = 2.4**2 / xdim
    if C0 is None:
        #C0 = np.diag((0.1*np.abs(x))**2)
        raise ValueError('Provide initial covariance')
    Ct = np.kron(np.eye(parallel_chains,dtype=np.uint32), C0)

    # Initialize chains
    index = np.arange(nsamples+warmup,dtype=np.uint32)
    chain_id = np.arange(parallel_chains,dtype=np.uint32)
    chain_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains,xdim),np.nan),dims=['sample','chain_id','param'],
                         coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id),param=(['param'],included)))
    chain_xr.loc[dict(sample=0)] = xt
    seed_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains),0,dtype=np.int32),dims=['sample','chain_id'],
                        coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id)))
    seed_xr.loc[dict(sample=0)] = init_config['seed'].iloc[0]
    
    constraints = ["ECS","TCR","Tinc","ERFari","ERFaci","ERFaer","CO2conc2022","OHC","Tvar"]
    constraints_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains,len(constraints)),np.nan),dims=['sample','chain_id','constraint'],
                               coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id),constraint=(['constraint'],constraints)))
    
    loss_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains),np.nan),dims=['sample','chain_id'],
                        coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id)))
    
    # Compute loss (negative log-likehood) from data, prior and constraints
    loss = compute_data_loss(model_T,obs,var,target=target_log_likelihood).rename({'config':'chain_id'})
    #data_loss_chain[0] = loss
    if default_priors:
        log_prior = get_log_prior(included)
        prior_loss = DataArray(data=-log_prior(xt),dims=['chain_id'],coords=dict(chain_id=(['chain_id'],chain_id)))
        loss += prior_loss
    constraints = compute_constraints(fair_run).squeeze(dim='scenario',drop=True).rename({'config':'chain_id'}).transpose('chain_id','constraint')
    if default_constraints:
        log_constraint_target = get_log_constraint_target()
        constraints_xr.loc[dict(sample=0)] = constraints
        constraint_loss = DataArray(data=-log_constraint_target(constraints),dims=['chain_id'],
                                    coords=dict(chain_id=(['chain_id'],chain_id)))
        loss += constraint_loss
    loss_xr.loc[dict(sample=0)] = loss
    
    # Create netcdf file for the results
    create_file(scenario,parallel_chains,included,warmup,filename=filename)
    
    start_time = time.perf_counter()
    t_start, t_end = 1, warmup+nsamples-1
    print('Warmup period...')
    for t in range(t_start,t_end+1):
        print(t)
        # Generate new proposal value for the chain
        proposal = sample_multivariate(xt.flatten(),Ct,newshape=(parallel_chains,xdim))
        #proposal = npr.multivariate_normal(xt.flatten(),Ct).reshape()
        configs[included] = proposal
        #seed = npr.randint(0,int(6e8),size=parallel_chains)
        configs['seed'] = 12345
        valid = validate_config(configs,bounds_dict=bounds,stochastic_run=False)
        fair_allocated = setup_fair([scenario],valid.sum(),start=1750,end=2023)
        fair_proposal = run_configs(fair_allocated,transform_df(configs[valid],transformation,name_changes=name_changes),
                                    solar_forcing,volcanic_forcing,start=1750,end=2023,stochastic_run=False)
        model_T = temperature_anomaly(fair_proposal,rolling_window_size=2).squeeze(dim='scenario',drop=True)
        loss_proposal = compute_data_loss(model_T,obs,var,target=target_log_likelihood).rename({'config':'chain_id'}).assign_coords({'chain_id': chain_id[valid]})
        if default_priors:
            #print(loss_proposal)
            #print(DataArray(data=-log_prior(proposal),dims=['chain_id'],coords=dict(chain_id=(['chain_id'],valid_indices))))
            loss_proposal += DataArray(data=-log_prior(proposal[valid]),
                                       dims=['chain_id'],coords=dict(chain_id=(['chain_id'],chain_id[valid])))
        proposal_constraints = compute_constraints(fair_proposal).squeeze(dim='scenario',drop=True).rename({'config':'chain_id'}).transpose(transpose_coords=False).assign_coords({'chain_id': chain_id[valid]})
        if default_constraints:
            loss_proposal += DataArray(data=-log_constraint_target(proposal_constraints),
                                       dims=['chain_id'],coords=dict(chain_id=(['chain_id'],chain_id[valid])))
        # Only valid configs have log_acceptance larger than -inf
        log_acceptance = np.full(parallel_chains,-np.inf)
        log_acceptance[valid] = -loss_proposal.data + loss.data[valid]
        # Accept or reject
        acc_arr = np.log(npr.uniform(0,1,size=parallel_chains)) < log_acceptance
        if acc_arr.sum() > 0:
            xt[acc_arr] = proposal[acc_arr]
            loss[acc_arr] = loss_proposal[acc_arr[valid]]
            constraints[acc_arr] = proposal_constraints[acc_arr[valid]]
            accepted[acc_arr] += np.ones(acc_arr.sum(),dtype=np.uint32)
        # Reset rejected configs
        configs.iloc[~acc_arr,:-1] = xt[~acc_arr]
        # Update xarrays
        chain_xr.loc[dict(sample=t)] = xt
        seed_xr.loc[dict(sample=t)] = configs['seed'].values
        loss_xr.loc[dict(sample=t)] = loss
        constraints_xr.loc[dict(sample=t)] = constraints
        
        # Adaptation starts after the warmup period
        if t == warmup - 1:
            print('...done')
            combined_samples = chain_xr.sel(sample=slice(warmup//2,warmup-1,thinning)).stack(dict(combined=['sample','chain_id']),create_index=False)
            Ct = np.kron(np.eye(parallel_chains,dtype=np.uint32),sd*np.cov(combined_samples.data,ddof=1))
            mean = np.repeat(combined_samples.mean(dim='combined').data[np.newaxis,:],parallel_chains,axis=0)
            print('Warmup acceptance ratio:',accepted.sum()/(warmup*parallel_chains))
            accepted[:] = 0
            print('Sampling posterior...')
            print(f'{progress}%')
            progress += 10
        elif t >= warmup:
            n = (t-warmup) + warmup//2 * parallel_chains // thinning
            # Recursive update for mean
            next_mean = 1/(n+1) * (n*mean + xt)
            # Recursive update for the covariance matrix
            Ct = (n-1)/n * Ct + sd/n * (n * parallel_outer_product(mean) - (n+1) * parallel_outer_product(next_mean) + parallel_outer_product(xt))
            # Update mean
            mean = next_mean
            # Print and save progress
            if (t-warmup) / (nsamples-1) * 100 >= progress:
                print(f'{progress}%')
                # Save results after each 10 %
                save_progress(chain_xr.sel(sample=slice(warmup,t,thinning)),
                              loss_xr.sel(sample=slice(warmup,t,thinning)),
                              seed_xr.sel(sample=slice(warmup,t,thinning)),
                              constraints_xr.sel(sample=slice(warmup,t,thinning)),
                              filename=filename)
                progress += 10
            
    print('...done')
    print('Acceptance ratios of parallel chains:',np.round(accepted/(nsamples*parallel_chains),3))
    end_time = time.perf_counter()
    print(f'Time taken: {(end_time-start_time):.1f} s')
    #print(f"AM performance:\nposterior samples = {nsamples}")

'''
def mcmc_extend(scenario,init_config,samples,bounds=None,use_constraints=True,use_prior=True,
                stochastic_run=False,Ct=None,folder='MC_results',filename='sampling',data_loss_method='wss'):
    #Extends the existing chains with provided number of samples.

    solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,1,1750,2101)
    gmst = read_gmst_temperature()
    T = read_hadcrut_temperature()
    if data_loss_method == 'wss':
        obs, var = gmst, T.std(dim='realization')**2
    elif data_loss_method == 'trend':
        trends = compute_trends(T)
        obs, var = np.mean(trends), np.std(trends)**2

    ds = Dataset(f'{cwd}/{folder}/{scenario}/{filename}.nc','r')
    params = ds['param'][:].tolist()
    xdim = ds.dimensions['param'].size
    sd = 2.4**2 / xdim
    chain, loss_chain = ds['chain'][:,:].data, ds['loss_chain'][:].data
    mean = chain.mean(axis=0).reshape((xdim,1))
    constraint_names = ['ECS','TCR','OHC',"T 2003-2022","ERFari","ERFaci","ERFaer", 'CO2', 'ohc', 'T 2081-2100']
    constraints_chain = np.column_stack(tuple(ds[constraint][:] for constraint in constraint_names))
    MAP = ds['MAP'][:].data
    if Ct is None:
        Ct = ds['Ct'][:,:].data
    warmup = int(ds['warmup'][:].data)
    seed_chain = ds['seeds'][:].data
    t_start, t_end = ds.dimensions['sample'].size, ds.dimensions['sample'].size + samples - 1
    ds.close()
    
    x = chain[-1,:]
    constraints = constraints_chain[-1,:]
    config = init_config.copy()
    config[params] = x
    config['seed'] = seed_chain[-1]
    loss = loss_chain[-1]
    # Run FaIR the first time
    fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                   start=1750,end=2101,stochastic_run=stochastic_run)
    # Temperature anomaly compared to temperature mean between year 1850 and 1900
    anomaly = fair.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0) -\
              fair.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds')
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
    
    if use_prior:
        log_prior = get_log_prior(params)
    constraints = compute_constraints(fair).squeeze()
    if use_constraints:
        log_constraint_target = get_log_constraint_target()
    
    proposal_config = MAP_config.copy()
    progress, accepted = 0, 0
    start_time = time.process_time()
    print('Sampling posterior...')
    for t in range(t_start,t_end+1):
        #proposal value for the chain
        proposal = npr.multivariate_normal(x,Ct)
        proposal_config[params] = proposal
        seed = npr.randint(0,int(6e8))
        proposal_config['seed'] = seed
        valid = validate_config(proposal_config,bounds=bounds)
        if not valid:
            log_acceptance = -np.inf
        else:
            fair_proposal = runFaIR(solar_forcing,volcanic_forcing,emissions,proposal_config,scenario,
                                    start=1750,end=2101,stochastic_run=stochastic_run)
            anomaly = fair_proposal.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0) -\
                      fair_proposal.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds')
            loss_proposal = compute_data_loss(anomaly,obs,var,data_loss_method=data_loss_method)
            if use_prior:
                loss_proposal += float(-log_prior(proposal.reshape((1,-1))))
            proposal_constraints = compute_constraints(fair_proposal)
            if use_constraints:
                loss_proposal += -float(log_constraint_target(constraints).squeeze())
            log_acceptance = -loss_proposal + loss
        #Accept or reject
        if np.log(npr.uniform(0,1)) <= log_acceptance:
            x = proposal
            if loss_proposal < loss_MAP:
                MAP = x
                loss_MAP = loss_proposal
            loss = loss_proposal
            constraints = proposal_constraints
            accepted += 1
        # Add new values to chains
        chain[t,:] = x
        seed_chain[t] = seed
        loss_chain[t] = loss
        constraints_chain[t,:] = constraints
        
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
            save_progress(scenario,warmup,Ct,chain[:(t+1),:],loss_chain[:(t+1)],seed_chain[:(t+1)],MAP,
                          constraints_arr=constraints_chain[:(t+1),:],filename=filename)
            progress += 10
    print('...done')
    end_time = time.process_time() 
    print(f"AM performance:\nposterior samples = {t_end+1-warmup}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")
'''

# For parallel sampling
def parallel_outer_product(X):
    shape = (1,len(X)) if X.ndim == 1 else X.shape
    temp = np.zeros((len(X),X.size))
    indices = (np.repeat(np.arange(shape[0],dtype=int),shape[-1]),
               np.arange(X.size,dtype=int))
    temp[indices] = X.flatten()
    return np.einsum('ij,ik->jk',temp,temp)

def sample_multivariate(mu,C,newshape=None,vectorized=True):
    if newshape is None:
        xdim = np.nonzero(C[:,0])[0].max() + 1
        newshape = (len(mu)//xdim,xdim)
    if vectorized:
        return npr.multivariate_normal(mu,C).reshape(newshape)
    else:
        return np.vstack(tuple(npr.multivariate_normal(mu[n*newshape[-1]:(n+1)*newshape[-1]],C[n*newshape[-1]:(n+1)*newshape[-1],n*newshape[-1]:(n+1)*newshape[-1]])
                               for n in range(newshape[0])))