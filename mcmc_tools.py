#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2023

@author: nurmelaj
"""

import numpy as np
import numpy.random as npr
import time
# Sampling and computation related functions from fair_tools
from fair_tools import setup_fair,run_configs,compute_data_loss,compute_constraints,temperature_anomaly,validate_config,compute_trends
from fair_tools import get_log_constraint_target,get_log_prior
from pandas import DataFrame,concat
from data_handling import read_forcing,read_temperature_obs,transform_df,create_file,save_progress,read_jumping_cov
from xarray import DataArray
from netCDF4 import Dataset
    
def mcmc_run(scenario,init_config,included,nsamples,warmup,C0=None,bounds=None,use_priors=True,use_constraints=True,
             use_Tvar=True,stochastic_run=False,target_log_likelihood='wss',filename='sampling',parallel_chains=10,thinning=10):
    # Read forcing data
    solar_forcing, volcanic_forcing = read_forcing(1750,2023)
    if target_log_likelihood == 'wss':
        obs, unc = read_temperature_obs()
        var = unc**2
    elif target_log_likelihood == 'trend_line':
        raise ValueError("Target log likelihood 'trend_line' not supported in current version")
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
    seeds = configs['seed'].values
    fair_run = run_configs(fair_allocated,transform_df(configs,transformation,name_changes=name_changes),
                           solar_forcing,volcanic_forcing,start=1750,end=2023,stochastic_run=stochastic_run)
    # Temperature anomaly compared to temperature mean between years 1850 and 1900
    model_T = temperature_anomaly(fair_run,rolling_window_size=2).squeeze(dim='scenario',drop=True)
    progress = 0
    accepted = np.zeros(parallel_chains,dtype=np.uint32)
    xdim = len(included)
    xt = configs[included].to_numpy()
    # Dimension related scaling for adaptive Metropolis algorithm
    sd = 2.4**2 / xdim
    if C0 is None:
        C0 = read_jumping_cov(included)
    Ct = np.kron(np.eye(parallel_chains,dtype=np.uint32),C0)

    # Initialize chains
    index = np.arange(nsamples+warmup,dtype=np.uint32)
    chain_id = np.arange(parallel_chains,dtype=np.uint32)
    chain_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains,xdim),np.nan),dims=['sample','chain_id','param'],
                         coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id),param=(['param'],included)))
    chain_xr.loc[dict(sample=0)] = xt
    seed_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains),0,dtype=np.uint32),dims=['sample','chain_id'],
                        coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id)))
    seed_xr.loc[dict(sample=0)] = seeds
    
    constraint_names = ['ECS','TCR','Tinc','ERFari','ERFaci','ERFaer','CO2conc2022','OHC']
    if use_Tvar:
        constraint_names.append('Tvar')
    constraints_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains,len(constraint_names)),np.nan),dims=['sample','chain_id','constraint'],
                               coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id),constraint=(['constraint'],constraint_names)))
    
    loss_xr = DataArray(data=np.full((warmup+nsamples,parallel_chains),np.nan),dims=['sample','chain_id'],
                        coords=dict(sample=(['sample'],index),chain_id=(['chain_id'],chain_id)))
    
    # Compute loss (negative log-likehood) from data, prior and constraints
    loss = compute_data_loss(model_T,obs,var,target=target_log_likelihood).rename({'config':'chain_id'})
    if use_priors:
        log_prior = get_log_prior(included)
        prior_loss = DataArray(data=-log_prior(xt),dims=['chain_id'],coords=dict(chain_id=(['chain_id'],chain_id)))
        loss += prior_loss
    constraints = compute_constraints(fair_run,constraints=constraint_names).squeeze(dim='scenario',drop=True).rename({'config':'chain_id'}).transpose('chain_id','constraint')
    if use_constraints:
        log_constraint_target = get_log_constraint_target(constraints=constraint_names)
        constraints_xr.loc[dict(sample=0)] = constraints
        constraint_loss = DataArray(data=-log_constraint_target(constraints),dims=['chain_id'],
                                    coords=dict(chain_id=(['chain_id'],chain_id)))
        loss += constraint_loss
    loss_xr.loc[dict(sample=0)] = loss
    
    # Create netcdf file for the results
    create_file(scenario,parallel_chains,included,constraint_names,warmup,filename=filename)
    
    start_time = time.perf_counter()
    print('Warmup period...')
    for t in range(1, warmup+nsamples):
        # Generate new proposal value for the chain
        proposal = sample_multivariate(xt.flatten(),Ct,newshape=(parallel_chains,xdim))
        proposal_seeds = npr.randint(0,int(6e8),size=parallel_chains)
        #proposal = npr.multivariate_normal(xt.flatten(),Ct).reshape()
        configs[included] = proposal
        configs['seed'] = proposal_seeds
        valid = validate_config(configs,bounds_dict=bounds,stochastic_run=stochastic_run)
        fair_allocated = setup_fair([scenario],valid.sum(),start=1750,end=2023)
        fair_proposal = run_configs(fair_allocated,transform_df(configs[valid],transformation,name_changes=name_changes),
                                    solar_forcing,volcanic_forcing,start=1750,end=2023,stochastic_run=stochastic_run)
        model_T = temperature_anomaly(fair_proposal,rolling_window_size=2).squeeze(dim='scenario',drop=True)
        loss_proposal = compute_data_loss(model_T,obs,var,target=target_log_likelihood).rename({'config':'chain_id'}).assign_coords({'chain_id': chain_id[valid]})
        if use_priors:
            loss_proposal += DataArray(data=-log_prior(proposal[valid]),
                                       dims=['chain_id'],coords=dict(chain_id=(['chain_id'],chain_id[valid])))
        proposal_constraints = compute_constraints(fair_proposal,constraints=constraint_names).squeeze(dim='scenario',drop=True).rename({'config':'chain_id'}).transpose(transpose_coords=False).assign_coords({'chain_id': chain_id[valid]})
        if use_constraints:
            loss_proposal += DataArray(data=-log_constraint_target(proposal_constraints),
                                       dims=['chain_id'],coords=dict(chain_id=(['chain_id'],chain_id[valid])))
        # Only valid configs have log_acceptance larger than -inf
        log_acceptance = np.full(parallel_chains,-np.inf)
        log_acceptance[valid] = -loss_proposal.data + loss.data[valid]
        # Accept or reject
        acc_arr = np.log(npr.uniform(0,1,size=parallel_chains)) < log_acceptance
        if acc_arr.sum() > 0:
            xt[acc_arr] = proposal[acc_arr]
            seeds[acc_arr] = proposal_seeds[acc_arr]
            loss[acc_arr] = loss_proposal[acc_arr[valid]]
            constraints[acc_arr] = proposal_constraints[acc_arr[valid]]
            accepted[acc_arr] += np.ones(acc_arr.sum(),dtype=np.uint32)
        # Reset rejected configs
        configs.iloc[~acc_arr,:] = np.column_stack((xt[~acc_arr], seeds[~acc_arr]))
        # Update xarrays
        chain_xr.loc[dict(sample=t)] = xt
        seed_xr.loc[dict(sample=t)] = seeds
        loss_xr.loc[dict(sample=t)] = loss
        constraints_xr.loc[dict(sample=t)] = constraints
        
        # Adaptation starts after the warmup period
        if t == warmup - 1:
            print('...done')
            combined_samples = chain_xr.sel(sample=slice(warmup//2,warmup-1,thinning)).stack(dict(combined=['sample','chain_id']),create_index=False)
            Ct = np.kron(np.eye(parallel_chains,dtype=np.uint32),sd*np.cov(combined_samples.data,ddof=1))
            mean = np.repeat(combined_samples.mean(dim='combined').data[np.newaxis,:],parallel_chains,axis=0)
            print(f'Warmup acceptance ratio: {(100*accepted.sum()/(warmup*parallel_chains)):.2f} %')
            accepted[:] = 0
            print('Sampling posterior...')
            print('0%')
            progress = 10
        elif t >= warmup and (t-warmup+1) % thinning == 0:
            n = (t-warmup) // thinning + warmup//2 * parallel_chains // thinning
            # Recursive update for mean
            next_mean = 1/(n+1) * (n*mean + xt)
            # Recursive update for the covariance matrix
            Ct = (n-1)/n * Ct + sd/n * (n * parallel_outer_product(mean) - (n+1) * parallel_outer_product(next_mean) + parallel_outer_product(xt))
            # Update mean
            mean = next_mean
            # Print and save progress
            if (t+1-warmup) / nsamples * 100 >= progress:
                print(f'{progress}%')
                print(chain_xr.sel(sample=slice(warmup,t,thinning)))
                # Save results after each 10 %
                save_progress(chain_xr.sel(sample=slice(warmup,t,thinning)),
                              loss_xr.sel(sample=slice(warmup,t,thinning)),
                              seed_xr.sel(sample=slice(warmup,t,thinning)),
                              constraints_xr.sel(sample=slice(warmup,t,thinning)),
                              filename=filename)
                progress += 10
    print('...done')
    print('Percentual acceptance ratios of parallel chains:',np.round(100*accepted/(nsamples*parallel_chains),2))
    end_time = time.perf_counter()
    print(f'Time taken: {(end_time-start_time):.1f} s')

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
    
def read_sampling(path):
    return Dataset(path,mode='r')

def read_sampling_configs(path,N=None):
    ds = read_sampling(path)
    Nsamples, parallel_chains, Nparams = ds.dimensions['sample'].size, ds.dimensions['chain_id'].size, ds.dimensions['param'].size
    param_samples = ds['chain'][:,:,:].reshape(((Nsamples*parallel_chains,Nparams)))
    N = min(Nsamples*parallel_chains,N) if N is not None else Nsamples*parallel_chains
    indices = np.arange(Nsamples*parallel_chains,dtype=np.uint32)
    picked = indices if N == Nsamples*parallel_chains else np.random.choice(indices,size=N,replace=False)
    df = DataFrame(data=param_samples[picked,:],columns=ds['param'][:])
    seeds = ds['seed'][:].reshape(((Nsamples*parallel_chains,)))
    df['seed'] = seeds[picked]
    ds.close()
    return df

def read_sampling_constraints(path,stochastic_rerun=False,N=None):
    ds = read_sampling(path)
    Nsamples, parallel_chains = ds.dimensions['sample'].size, ds.dimensions['chain_id'].size
    N = min(Nsamples*parallel_chains,N) if N is not None else Nsamples*parallel_chains
    indices = np.arange(Nsamples*parallel_chains,dtype=np.uint32)
    picked = indices if N == Nsamples*parallel_chains else np.random.choice(indices,size=N,replace=False)
    if stochastic_rerun:
        solar_forcing, volcanic_forcing = read_forcing(1750,2100)
        fair_allocated = setup_fair([ds.scenario],N,start=1750,end=2100)
        Nparams = ds.dimensions['param'].size
        config_samples = ds['chain'][:,:,:].data.reshape((Nsamples*parallel_chains,Nparams))
        seeds = ds['seed'][:].reshape(((Nsamples*parallel_chains,)))
        configs_df = transform_df(DataFrame(data=config_samples[picked,:],columns=ds['param'][:]),
                                  {'log(-aci_beta)': lambda x: -np.exp(x),
                                   'log(aci_shape_SO2)': lambda x: np.exp(x),
                                   'log(aci_shape_BC)': lambda x: np.exp(x),
                                   'log(aci_shape_OC)': lambda x: np.exp(x)},
                                  {'log(-aci_beta)':'aci_beta',
                                   'log(aci_shape_SO2)':'aci_shape_SO2',
                                   'log(aci_shape_BC)':'aci_shape_BC',
                                   'log(aci_shape_OC)':'aci_shape_OC'})
        configs_df['seed'] = seeds[picked]
        fair_rerun = run_configs(fair_allocated,configs_df,solar_forcing,volcanic_forcing,start=1750,end=2100,stochastic_run=True)
        constraints = compute_constraints(fair_rerun,constraints=ds['constraint'][:])
        df = DataFrame(data=constraints.sel(scenario=ds.scenario)[:,:].data.T,columns=ds['constraint'][:])
    else:
        Nconstraints = ds.dimensions['constraint'].size
        constraint_samples = ds['constraints'][:,:,:].data.reshape((Nsamples*parallel_chains,Nconstraints))
        df = DataFrame(data=constraint_samples[picked,:],columns=ds['constraint'][:])
    ds.close()
    return df
    