#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:39:15 2023

@author: nurmelaj
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 07:22:08 2023
​
@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
cwd = os.getcwd()

from fair_tools import load_data, load_configs, load_MC_configs, runFaIR
from pickle_tools import save_obj
from scipy.stats import gamma, norm, uniform
from netCDF4 import Dataset
from dotenv import load_dotenv

import time
from fitter import Fitter

load_dotenv()
figdir=f"{cwd}/figures"
#fair_calibration_dir=f"{cwd}/fair-calibrate"

# Choose scenario
#scenario = 'ssp245'
scenario = 'ssp245'
start = 1850
end = 2020
# MC parameters
samples = 2000
warmup = 100

df_configs = load_configs()
ignored = ['ari BC', 'ari CH4', 'ari N2O', 'ari NH3', 'ari NOx', 'ari OC', 'ari Sulfur', 'ari VOC',
           'ari Equivalent effective stratospheric chlorine','seed']
included = [param for param in df_configs.columns if param not in ignored]

# Piror density functions
distributions = []
params = []
for col in df_configs.columns:
    fit = Fitter(df_configs[col],distributions=["gamma","norm","uniform"])
    fit.fit(progress=False)
    best = fit.get_best(method = 'sumsquare_error')
    for name in best.keys():
        distributions.append(name)
        params.append(tuple(best[name].values()))
distributions = np.array(distributions)
is_gamma = distributions == 'gamma'
is_norm = distributions == 'norm'
is_uniform = distributions == 'uniform'
gamma_params = np.array([params[i] for i in np.where(is_gamma)[0]])
norm_params = np.array([params[i] for i in np.where(is_norm)[0]])
uniform_params = np.array([params[i] for i in np.where(is_uniform)[0]])

prior = lambda x: np.prod(gamma.pdf(x[is_gamma[~is_uniform]],gamma_params[:,0],gamma_params[:,1],gamma_params[:,2])) * np.prod(norm.pdf(x[is_norm[~is_uniform]],norm_params[:,0],norm_params[:,1]))

solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,end)
# Temperature data
ds_mean = Dataset(f'{cwd}/fair-calibrate/data/HadCrut5_mean.nc')
years = ds_mean['year'][:].data
T_data = ds_mean['tas'][:].data
ds_std = Dataset(f'{cwd}/fair-calibrate/data/HadCrut5_std.nc')
T_std = ds_std['tas'][:].data
proposed_config = df_configs.mean(axis=0).copy().to_frame().transpose()
prior_configs = proposed_config.copy()
fair_prior = runFaIR(solar_forcing,volcanic_forcing,emissions,prior_configs,scenario,end=end)
T_prior = fair_prior.temperature.loc[dict(scenario=scenario,layer=0)].mean(axis=1)

def residual(config):
    fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                   start=1750,end=2020)
    T_model = fair.temperature.loc[dict(scenario=scenario,layer=0)].mean(axis=1)
    return T_model[(start-1750):] - T_data

def validate_config(config):
    #These åaramers require postivity
    positive = ['gamma', 'c1', 'c2', 'c3', 'kappa1', 'kappa2', 'kappa3', 
                'epsilon', 'sigma_eta', 'sigma_xi']
    valid = (config[positive] > 0).all(axis=1) & (config['gamma'] > 0.8) & (config['c1'] > 2) & \
            (config['c2'] > config['c1']) & (config['c3'] > config['c2']) & (config['kappa1'] > 0.3)
    return valid[0]

def mcmc_run(residual,x0,samples,warmup,C0=None,std=None,prior=None):
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2020)
    var = std**2
    xdim = len(x0)
    sd = 2.4**2 / xdim
    proposed_config[included] = x0
    wss = np.sum(np.square(residual(proposed_config))/var)
    log_pos = -0.5 * wss - 0.5 * np.sum(np.log(var)) + np.log(prior(x0))
    # Percentual progress
    progress = 0
    # Acceptance ratio
    accepted = 0
    # Initialize chains
    chain = np.full((warmup+samples,xdim),np.nan)
    chain[0,:] = x0
    ss_chain = np.full((warmup+samples,xdim),np.nan)
    ss_chain[0] = wss
    mean = chain[0,:].reshape((xdim,1))
    Ct = C0
    start_time = time.process_time()
    for t in range(1,warmup+samples):
        if t / (samples+warmup-1) * 100 >= progress:
            print(f'{progress}%')
            progress += 10
        #Proposed value for the chain
        proposed = np.random.multivariate_normal(chain[t-1,:],Ct)
        proposed_config[included] = proposed
        valid = validate_config(proposed_config)
        if not valid:
            log_acceptance = -np.inf
        else:
            wss_proposed = np.sum(np.square(residual(proposed_config))/var)
            proposed_log_pos = -0.5 * wss_proposed - 0.5 * np.sum(np.log(var)) + np.log(prior(proposed))
            log_acceptance = proposed_log_pos - log_pos
        #Accept or reject
        if np.log(np.random.uniform(0,1)) <= log_acceptance:
            chain[t,:] = proposed
            accepted += 1
            log_pos = proposed_log_pos
            wss = wss_proposed
            #print('accepted')
        else:
            chain[t,:] = chain[t-1,:]
            proposed_config[included] = chain[t-1,:]
        ss_chain[t] = wss
            #print('rejected')
        #Value in chain as column vector
        vec = chain[t,:].reshape((xdim,1))
        #Recursive update for mean
        next_mean = 1/(t+1) * (t*mean + vec)
        #Adaptation starts after the warmup period
        if t >= warmup - 1:
            if (t+1-warmup) % 10000 == 0 or t+1 == warmup+samples:
                np.save(f'MC_results/{scenario}_samples={t+1-warmup}',chain[warmup:(t-1),:])
                np.save(f'MC_results/{scenario}_cov_{t+1-warmup}',Ct)
            # Recursive update for the covariance matrix
            epsilon = 0.0
            Ct = (t-1)/t * Ct + sd/t * (t * mean @ mean.T - (t+1) * next_mean @ next_mean.T + vec @ vec.T + epsilon*np.eye(xdim))
        #Update mean
        mean = next_mean
    #Empirical mean and covariance in the chain after the warmup
    mu = np.mean(chain[warmup:,:],axis=0)
    cov = np.cov(chain[warmup:,:],rowvar=False)
    end_time = time.process_time()
    
    print(f"MH performance:\nposterior samples = {samples}\ntime: {end_time-start_time} s\nacceptance ratio = {100*accepted/samples:.2f} %")
    return {'chain':chain,'ss_chain':ss_chain,'samples':samples,'warmup':warmup,'mu':mu,'cov':cov,'names':included}

def plot_distributions(df_configs,mu,cov,chain):
    fig, axs = plt.subplots(7,7,figsize=(25,20))
    fig.canvas.set_window_title('Distributions')
    fig.tight_layout(pad=5)
    plt.ticklabel_format(useOffset=False)
    col_names = df_configs.columns
    stds = np.sqrt(np.diag(cov))
    i = 0
    for n in range(len(col_names)):
        name = col_names[n]
        row, col = (n-n%7)//7, n%7
        data = df_configs[name]
        axs[row][col].hist(data,bins=20,density=True)
        axs[row][col].set_title(name)
        xmin, xmax = np.min(data), np.max(data)
        x = np.linspace(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin),100)
        if distributions[n] == 'gamma':
            axs[row][col].plot(x,gamma.pdf(x,*params[n]),linestyle='--',color='black')
            axs[row][col].hist(chain[warmup:,i],bins=20,density=True)
            axs[row][col].plot(x,norm.pdf(x,mu[i],stds[i]),linestyle='-',color='black')
            i += 1
        elif distributions[n] == 'norm':
            axs[row][col].plot(x,norm.pdf(x,*params[n]),linestyle='--',color='black')
            axs[row][col].hist(chain[warmup:,i],bins=20,density=True)
            axs[row][col].plot(x,norm.pdf(x,mu[i],stds[i]),linestyle='-',color='black')
            i += 1
        else:
            axs[row][col].plot(x,uniform.pdf(x,*params[n]),linestyle='--',color='black')
    for n in range(len(col_names),7*7):
        row, col = (n-n%7)//7, n%7
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    fig.savefig('plots/distributions')
    plt.show()
    
def plot_temperature(prior,posterior,data,std,start,end):
    years = list(range(start,end+1))
    T_fig = plt.figure('T',figsize=(10,5))
    ax_T = T_fig.gca()
    ax_T.plot(years, prior[(start-1750):], linestyle='--', color='black', markersize=2,label='prior')
    ax_T.plot(years, posterior[(start-1750):], linestyle='-', color='black', markersize=2,label='posterior')
    ax_T.errorbar(range(1850,2021), y=data, yerr=std, fmt='.',capsize=5,markersize=1,color='orange')
    ax_T.plot(range(1850,2021), data, 'r.', markersize=3)
    ax_T.set_title(f'{scenario}: temperature trend')
    ax_T.set_xlabel('year')
    ax_T.set_ylabel('Temperature (°C)')
    ax_T.legend()
    T_fig.savefig(f'plots/{scenario}_trend')
    plt.show()

x0 = proposed_config[included].to_numpy().squeeze()
#stds = df_configs[included].std(axis=0).to_numpy()
C0 = np.diag((0.1*df_configs[included].std(axis=0).to_numpy())**2)

filename = f'sampling_{scenario}'
sampling = mcmc_run(residual,x0,samples,warmup,C0=C0,std=T_std,prior=prior)
save_obj(sampling, 'MC_results/', filename)
#mu, cov, chain = sampling['mu'], sampling['cov'], sampling['chain']
#warmup = sampling['warmup']
sampled_configs = load_MC_configs('MC_results/',scenario,included)

'''
fair_posterior = runFaIR(solar_forcing,volcanic_forcing,emissions,sampled_configs,scenario,end=end)
T_post = fair_posterior.temperature.loc[dict(scenario=scenario,layer=0)].mean(axis=1)
plot_distributions(sampled_configs,mu,cov,chain)
'''
'''
specie = 'CO2'
specie_fig = plt.figure('specie')
ax_specie = specie_fig.gca()
T_fig = plt.figure('T')

ax_specie.plot(f.timebounds, f.concentration.loc[dict(scenario=scenario, specie=specie)], label=f.configs)
ax_specie.set_title(f'{scenario}: {specie} concentration')
ax_specie.set_xlabel('year')
ax_specie.set_ylabel(f'{specie} concentration')
'''
#plot_temperature(T_prior, T_post, T_data, T_std, start, end)
