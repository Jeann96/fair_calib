#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:18:44 2023

@author: nurmelaj
"""

import numpy as np
import time
from fair_tools import load_data, runFaIR, read_temperature_data, load_configs
from netCDF4 import Dataset
import os
cwd = os.getcwd()

def validate_config(config,bounds=None):
    #These parameters require postivity and other constraints
    positive = ['gamma', 'c1', 'c2', 'c3', 'kappa1', 'kappa2', 'kappa3', 
                'epsilon', 'sigma_eta', 'sigma_xi']
    valid = ((config[positive] >= 0).all(axis=1) & (config['gamma'] > 0.8) & (config['c1'] > 2) & \
            (config['c2'] > config['c1']) & (config['c3'] > config['c2']) & (config['kappa1'] > 0.3))[0]
    if bounds is not None:
        in_bounds = all((min(bounds[param]) <= config[param].iloc[0]) & (config[param].iloc[0] <= max(bounds[param])) for param in bounds.keys())
    else:
        in_bounds = True
    return valid & in_bounds

def mcmc_run(scenario,x0,names,samples,warmup,C0=None,prior=None,bounds=None,
             extend=False,folder='MC_results'):
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2020)
    data, std = read_temperature_data()
    proposed_config = load_configs().mean(axis=0).to_frame().transpose()
    def residual(config):
        fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                       start=1750,end=2020)
        model = fair.temperature.loc[dict(scenario=scenario,layer=0)].mean(axis=1)
        return model[100:] - data
    var = std**2
    xdim = len(x0)
    sd = 2.4**2 / xdim
    proposed_config[names] = x0
    wss = np.sum(np.square(residual(proposed_config))/var)
    # Negative log-likelihood of the initial value (loss function)
    loss = 0.5 * wss + 0.5 * np.sum(np.log(var)) - np.log(prior(x0))
    # MAP (maximum posterior) estimate, equal to minimum of the loss function
    MAP = x0
    # Percentual progress
    progress = 0
    # Acceptance ratio
    accepted = 0
    if not extend:
        create_file(scenario,names,warmup)
        Ct = C0
        # Initialize chains
        chain = np.full((warmup+samples,xdim),np.nan)
        chain[0,:] = x0
        ss_chain = np.full(warmup+samples,np.nan)
        ss_chain[0] = wss
        mean = chain[0,:].reshape((xdim,1))
        t_start, t_end = 1, warmup+samples
    else:
        ds = Dataset(f'{cwd}/{folder}/{scenario}/sampling.nc','a')
        mean = ds['mut'][:].reshape((xdim,1))
        Ct = ds['Ct'][:,:].data
        chain = ds['chain'][:,:].data
        ss_chain =  ds['ss_chain'][:].data
        size = ds.dimensions['sample'].size
        ds.close()
        t_start, t_end = size, size + samples
        # Ectend chains
        chain = np.append(chain, np.full((samples,xdim),np.nan), axis=0)
        ss_chain = np.append(ss_chain, np.full(samples,np.nan), axis=0)
    start_time = time.process_time()
    for t in range(t_start,t_end):
        #Proposed value for the chain
        proposed = np.random.multivariate_normal(chain[t-1,:],Ct)
        proposed_config[names] = proposed
        valid = validate_config(proposed_config,bounds=bounds)
        if not valid:
            log_acceptance = -np.inf
        else:
            wss_proposed = np.sum(np.square(residual(proposed_config))/var)
            # Negative log-likelihood (loss function)
            proposed_loss = 0.5 * wss_proposed + 0.5 * np.sum(np.log(var)) - np.log(prior(proposed))
            log_acceptance = -proposed_loss + loss
        #Accept or reject
        if np.log(np.random.uniform(0,1)) <= log_acceptance:
            chain[t,:] = proposed
            MAP = proposed if proposed_loss < loss else MAP
            loss = proposed_loss
            wss = wss_proposed
            accepted += 1
        else:
            chain[t,:] = chain[t-1,:]
            proposed_config[names] = chain[t-1,:]
        ss_chain[t] = wss
        #Value in chain as column vector
        vec = chain[t,:].reshape((xdim,1))
        #Recursive update for mean
        next_mean = 1/(t+1) * (t*mean + vec)
        #Adaptation starts after the warmup period
        if t >= warmup - 1:
            # Recursive update for the covariance matrix
            epsilon = 0.0
            Ct = (t-1)/t * Ct + sd/t * (t * mean @ mean.T - (t+1) * next_mean @ next_mean.T + vec @ vec.T + epsilon*np.eye(xdim))
        #Update mean
        mean = next_mean
        # Print and save progress
        if (t+1) / t_end * 100 >= progress:
            print(f'{progress}%')
            # Save results after each 10 %
            if progress != 0:
                save_progress(scenario,warmup,chain[:(t+1),:],ss_chain[:(t+1)],mean.flatten(),Ct,MAP)
            progress += 10
    end_time = time.process_time() 
    print(f"MH performance:\nposterior samples = {max(t_end-warmup,0)}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")

def create_file(scenario,params,warmup):
    N = len(params)
    ncfile = Dataset(f'MC_results/{scenario}/sampling.nc',mode='w')
    ncfile.createDimension('sample', None)
    ncfile.createDimension('param', N)
    ncfile.title = f'FaIR MC sampling for scenario {scenario}'
    ncfile.createVariable('sample', int, ('sample',),fill_value=False)
    ncfile.createVariable('param', str, ('param',),fill_value=False)
    ncfile['param'][:] = np.array(params)
    ncfile.createVariable('chain',float,('sample','param'),fill_value=False)
    ncfile.createVariable('ss_chain',float,('sample',),fill_value=False)
    ncfile.createVariable('MAP',float,('param',),fill_value=False)
    ncfile.createVariable('cov',float,('param','param'),fill_value=False)
    ncfile.createVariable('Ct',float,('param','param'),fill_value=False)
    ncfile.createVariable('mut',float,('param',),fill_value=False)
    ncfile.createVariable('warmup',int,fill_value=False)
    ncfile['warmup'][:] = warmup
    ncfile.close()

def save_progress(scenario,warmup,chain,ss_chain,mu_t,C_t,MAP,folder='MC_results'):
    t = len(chain) - 1
    xdim = chain.shape[1]
    ncfile = Dataset(f'{folder}/{scenario}/sampling.nc',mode='a')
    index = ncfile['sample'][:]
    if len(index) == 0:
        ncfile['sample'][:] = np.arange(0,t+1,1,dtype=int)
        ncfile['chain'][0:(t+1),:] = chain
        ncfile['ss_chain'][0:(t+1)] = ss_chain
    else:
        ncfile['sample'][:] = np.append(index,np.arange(index[-1]+1,t+1,1,dtype=int))
        ncfile['chain'][(index[-1]+1):(t+1),:] = chain[(index[-1]+1):,:]
        ncfile['ss_chain'][(index[-1]+1):(t+1)] = ss_chain[(index[-1]+1):]
    ncfile['Ct'][:,:] = C_t
    ncfile['mut'][:] = mu_t
    if t+1 <= warmup:
        ncfile['MAP'][:] = np.full(xdim,np.nan)
        ncfile['cov'][:] = np.full((xdim,xdim),np.nan)
    else:
        ncfile['MAP'][:] = MAP
        ncfile['cov'][:] = np.cov(chain[warmup:,:],rowvar=False)
    ncfile.close()
    
    