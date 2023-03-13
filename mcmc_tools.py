#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:18:44 2023

@author: nurmelaj
"""

import numpy as np
import time
from fair_tools import load_data, runFaIR, read_temperature_data
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

def mcmc_run(scenario,init_config,names,samples,warmup,C0=None,prior=None,bounds=None,
             folder='MC_results'):
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2020)
    data, std = read_temperature_data()
    def residual(config):
        fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                       start=1750,end=2020)
        model = fair.temperature.loc[dict(timebounds=slice(1850,2020),scenario=scenario,layer=0)].mean(axis=1)
        return model.to_numpy() - data.to_numpy()
    # Observation variance
    var = std.to_numpy()**2
    # Proposed fair configuration
    proposed_config = init_config.copy()
    progress, accepted = 0, 0
    x0 = init_config[names].to_numpy().squeeze()
    if C0 is None:
        C0 = np.diag((0.1*x0)**2)
    create_file(scenario,names,warmup,C0)
    xdim = len(x0)
    wss = np.sum(np.square(residual(init_config))/var)
    # MAP (maximum posterior) estimate, equal to minimum of the loss function
    MAP = x0
    # Negative log-likelihood of the initial value (loss function)
    loss = 0.5 * wss + 0.5 * np.sum(np.log(2*np.pi*var)) - np.log(prior(x0))
    Ct = C0
    # Initialize chains
    chain = np.full((warmup+samples,xdim),np.nan)
    chain[0,:] = x0
    wss_chain = np.full(warmup+samples,np.nan)
    wss_chain[0] = wss
    loss_chain = np.full(warmup+samples,np.nan)
    loss_chain[0] = loss
    t_start, t_end = 1, warmup+samples
    start_time = time.process_time()
    # Dimension related scaling parameter
    sd = 2.4**2 / xdim
    for t in range(t_start,t_end):
        if t == 1:
            print('Warmup period...')
        elif t == warmup:
            print('...done')
            print('Sampling posterior...')
        #Proposed value for the chain
        proposed = np.random.multivariate_normal(chain[t-1,:],Ct)
        proposed_config[names] = proposed
        valid = validate_config(proposed_config,bounds=bounds)
        if not valid:
            log_acceptance = -np.inf
        else:
            wss_proposed = np.sum(np.square(residual(proposed_config))/var)
            # Negative log-likelihood (loss function)
            proposed_loss = 0.5 * wss_proposed + 0.5 * np.sum(np.log(2*np.pi*var)) - np.log(prior(proposed))
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
        wss_chain[t] = wss
        loss_chain[t] = loss
        #Adaptation starts after the warmup period
        if t >= warmup:
            #Value in chain as column vector
            vec = chain[t,:].reshape((xdim,1))
            # First mean is the current value (after the warmup period)
            if t == warmup:
                mean = chain[t-1,:].reshape((xdim,1))
            #Recursive update for mean
            next_mean = 1/(t-warmup+2) * ((t-warmup+1)*mean + vec)
            # Recursive update for the covariance matrix
            if t == warmup:
                Ct = C0
            else:
                Ct = (t-warmup)/(t-warmup+1) * Ct + sd/(t-warmup+1) * ((t-warmup+1) * mean @ mean.T - (t-warmup+2) * next_mean @ next_mean.T + vec @ vec.T)
            #Update mean
            mean = next_mean
        # Print and save progress
        if (t-t_start-warmup+1) / (t_end-t_start-warmup) * 100 >= progress:
            print(f'{progress}%')
            # Save results after each 10 %
            save_progress(scenario,warmup,chain[:(t+1),:],wss_chain[:(t+1)],loss_chain[:(t+1)],MAP)
            progress += 10
    print('...done')
    end_time = time.process_time() 
    print(f"AM performance:\nposterior samples = {max(t_end-warmup,0)}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")
    
def mcmc_extend(scenario, samples, MAP_config, prior=None, bounds=None, folder='MC_results'):
    '''
    Extends the existing chains with provided number of samples.
    '''
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2020)
    data, std = read_temperature_data()
    def residual(config):
        fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                       start=1750,end=2020)
        model = fair.temperature.loc[dict(timebounds=slice(1850,2020),scenario=scenario,layer=0)].mean(axis=1)
        return model.to_numpy() - data.to_numpy()
    # Observation variance
    var = std.to_numpy()**2
    ds = Dataset(f'{cwd}/{folder}/{scenario}/sampling.nc','r')
    xdim = ds.dimensions['param'].size
    sd = 2.4**2 / xdim
    params = ds['param'][:]
    chain = ds['chain'][:,:].data
    warmup = int(ds['warmup'][:].data)
    MAP = ds['MAP'][:]
    wss_chain = ds['wss_chain'][:].data
    loss_chain = ds['loss_chain'][:].data
    wss = wss_chain[-1]
    loss = loss_chain[-1]
    t_start, t_end = ds.dimensions['sample'].size, ds.dimensions['sample'].size + samples
    mean = chain[warmup-1,:].reshape((xdim,1))
    for t in range(warmup,t_start):
        vec = chain[t,:].reshape((xdim,1))
        next_mean = 1/(t-warmup+2) * ((t-warmup+1)*mean + vec)
        if t == warmup:
            Ct = ds['C0'][:,:]
        else:
            Ct = (t-warmup)/(t-warmup+1) * Ct + sd/(t-warmup+1) * ((t-warmup+1) * mean @ mean.T - (t-warmup+2) * next_mean @ next_mean.T + vec @ vec.T)
        mean = next_mean
    ds.close()
    # Extend chains
    chain = np.append(chain, np.full((samples,xdim),np.nan), axis=0)
    wss_chain = np.append(wss_chain, np.full(samples,np.nan), axis=0)
    loss_chain = np.append(loss_chain, np.full(samples,np.nan), axis=0)
    proposed_config = MAP_config.copy()
    progress, accepted = 0, 0
    start_time = time.process_time()
    print('Sampling posterior...')
    for t in range(t_start,t_end):
        #Proposed value for the chain    Ct = 
        proposed = np.random.multivariate_normal(chain[t-1,:],Ct)
        proposed_config[params] = proposed
        valid = validate_config(proposed_config,bounds=bounds)
        if not valid:
            log_acceptance = -np.inf
        else:
            wss_proposed = np.sum(np.square(residual(proposed_config))/var)
            # Negative log-likelihood (loss function)
            proposed_loss = 0.5 * wss_proposed + 0.5 * np.sum(np.log(2*np.pi*var)) - np.log(prior(proposed))
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
            proposed_config[params] = chain[t-1,:]
        wss_chain[t] = wss
        loss_chain[t] = loss
        #Value in chain as column vector
        vec = chain[t,:].reshape((xdim,1))
        #Recursive update for mean
        next_mean = 1/(t-warmup+2) * ((t-warmup+1)*mean + vec)
        # Recursive update for the covariance matrix
        Ct = (t-warmup)/(t-warmup+1) * Ct + sd/(t-warmup+1) * ((t-warmup+1) * mean @ mean.T - (t-warmup+2) * next_mean @ next_mean.T + vec @ vec.T)
        #Update mean
        mean = next_mean
        # Print and save progress
        if (t-t_start+1) / (t_end-t_start) * 100 >= progress:
            print(f'{progress}%')
            # Save results after each 10 %
            save_progress(scenario,warmup,chain[:(t+1),:],wss_chain[:(t+1)],loss_chain[:(t+1)],MAP)
            progress += 10
    print('...done')
    end_time = time.process_time() 
    print(f"AM performance:\nposterior samples = {max(t_end-warmup,0)}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")

def create_file(scenario,params,warmup,C0):
    N = len(params)
    ncfile = Dataset(f'MC_results/{scenario}/sampling.nc',mode='w')
    ncfile.createDimension('sample', None)
    ncfile.createDimension('param', N)
    ncfile.title = f'FaIR MC sampling for scenario {scenario}'
    ncfile.createVariable('sample', int, ('sample',),fill_value=False)
    ncfile.createVariable('param', str, ('param',),fill_value=False)
    ncfile['param'][:] = np.array(params)
    ncfile.createVariable('chain',float,('sample','param'),fill_value=False)
    ncfile.createVariable('wss_chain',float,('sample',),fill_value=False)
    ncfile.createVariable('loss_chain',float,('sample',),fill_value=False)
    ncfile.createVariable('MAP',float,('param',),fill_value=False)
    ncfile.createVariable('cov',float,('param','param'),fill_value=False)
    ncfile.createVariable('C0',float,('param','param'),fill_value=False)
    ncfile['C0'][:,:] = C0
    ncfile.createVariable('warmup',int,fill_value=False)
    ncfile['warmup'][:] = warmup
    ncfile.close()

def save_progress(scenario,warmup,chain,wss_chain,loss_chain,MAP,folder='MC_results'):
    N = len(chain)
    ncfile = Dataset(f'{folder}/{scenario}/sampling.nc',mode='a')
    index = ncfile['sample'][:]
    if len(index) == 0:
        ncfile['sample'][:] = np.arange(0,N,1,dtype=int)
        ncfile['chain'][0:,:] = chain
        ncfile['wss_chain'][0:N] = wss_chain
        ncfile['loss_chain'][0:N] = loss_chain
    else:
        ncfile['sample'][:] = np.append(index,np.arange(index[-1]+1,N,1,dtype=int))
        ncfile['chain'][(index[-1]+1):N,:] = chain[(index[-1]+1):,:]
        ncfile['wss_chain'][(index[-1]+1):N] = wss_chain[(index[-1]+1):]
        ncfile['loss_chain'][(index[-1]+1):N] = loss_chain[(index[-1]+1):]
    ncfile['MAP'][:] = MAP
    ncfile['cov'][:] = np.cov(chain[warmup:,:],rowvar=False)
    ncfile.close()
    
    