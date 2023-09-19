#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2023

@author: nurmelaj
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm, uniform, gaussian_kde
from fair_tools import runFaIR,load_data,read_temperature_data,compute_data_loss,compute_prior_loss,compute_constrained_loss,compute_constraints

def plot_distributions(prior_configs,pos_configs,exclude=[],distributions=None):
    fig, axs = plt.subplots(7,7,figsize=(25,20))
    #fig.canvas.set_window_title('Distributions')
    fig.tight_layout(pad=5)
    plt.ticklabel_format(useOffset=False)
    col_names = prior_configs.columns
    for n in range(len(col_names)):
        name = col_names[n]
        row, col = (n-n%7)//7, n%7
        data = prior_configs[name].values
        axs[row][col].hist(data,bins=100,density=True)
        if name not in exclude:
            axs[row][col].hist(pos_configs[name],bins=100,density=True)
        axs[row][col].set_title(name)
        #if name == 'solar_trend':
        #    axs[row][col].set_ylim(0,150)
        #    axs[row][col].set_xlim(-0.075,0.075)
        xmin, xmax = np.min(data), np.max(data)
        if distributions is not None:
            x = np.linspace(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin),100)
            if distributions[name]['distribution'] == 'gamma':
                axs[row][col].plot(x,gamma.pdf(x,*distributions[name]['params']),linestyle='--',color='black')
            elif distributions[name]['distribution'] == 'norm':
                axs[row][col].plot(x,norm.pdf(x,*distributions[name]['params']),linestyle='--',color='black')
            elif distributions[name]['distribution'] == 'uniform':
                axs[row][col].plot(x,uniform.pdf(x,*distributions[name]['params']),linestyle='--',color='black')
            else:
                raise ValueError(f'Unknown distribution: {name}')
        
    for n in range(len(col_names),7*7):
        row, col = (n-n%7)//7, n%7
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    #fig.savefig('plots/distributions')
    plt.show() 
    
def layout_shape(N,threshold=3):
    if np.sqrt(N).is_integer():
        return (int(np.sqrt(N)),int(np.sqrt(N)))
    elif N in [2,3,5]:
        return (N,1)
    else:
        while True:
            s = int(np.ceil(np.sqrt(N)))
            a = s
            while a - s <= threshold:
                for b in range(a-threshold,a):
                    if a*b == N:
                        return (a,b)
                a += 1
            N += 1   
    
def plot_chains(chain,params=None):
    if params is None:
        params = [f'x_{n+1}' for n in range(chain.shape[1])]
    if len(params) != chain.shape[1]:
        raise ValueError(f'Length of parameter names {len(params)} differs from chain dimension {chain.shape[1]}')
    samples = len(chain)
    shape = layout_shape(len(params))
    fig, axs = plt.subplots(shape[1],shape[0],figsize=(30,25))
    #fig.canvas.set_window_title('Distributions')
    fig.tight_layout(pad=5)
    plt.ticklabel_format(useOffset=False)
    for n in range(len(params)):
        name = params[n]
        row, col = (n-n % shape[0])//shape[0], n % shape[0]
        axs[row][col].plot(chain[:,n],linestyle='-',color='black',linewidth=0.5)
        axs[row][col].set_xticks(ticks=[0,samples//4],labels=[])
        axs[row][col].set_title(name)
    for n in range(len(params),shape[0]*shape[1]):
        row, col = (n-n % shape[0])//shape[0], n % shape[0]
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    plt.show()
    
def plot_loss_test(config,params,param_ranges,prior=None,targets=None,model_var=None,
                   scenario='ssp245',N=100,ylims=None):
    solar_forcing, volcanic_forcing, emissions = load_data(scenario,1,1750,2100)
    data, std = read_temperature_data()
    data, var = data.data, std.data**2
    if model_var is not None:
        var += model_var
    shape = layout_shape(len(params))
    fig, axs = plt.subplots(shape[1],shape[0],figsize=(40,25))
    #fig.canvas.set_window_title('Distributions')
    fig.tight_layout(pad=5)
    plt.ticklabel_format(useOffset=False)
    if ylims is None:
        ylims_dict = {}
    for n in range(len(params)):
        name = params[n]
        orig_val = float(config[name])
        print(f'Param: {name}')
        vals = np.linspace(min(param_ranges[name]),max(param_ranges[name]),N)
        losses = []
        for param_val in vals:
            config[name] = param_val
            fair = runFaIR(solar_forcing,volcanic_forcing,emissions,config,scenario,
                           start=1750,end=2100)
            # Temperature anomaly compared to temperature mean between year 1850 and 1900
            anomaly = fair.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0).to_numpy().squeeze() \
                    - fair.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
            # Compute loss (negative log-likehood) from data, prior and constraints
            loss = compute_data_loss(anomaly, data, var)
            if prior is not None:
                loss += compute_prior_loss(prior, config[params].to_numpy().squeeze())
            if targets is not None:
                loss += compute_constrained_loss(compute_constraints(fair),targets)
            losses.append(loss)
        row, col = (n-n % shape[0])//shape[0], n % shape[0]
        axs[row][col].set_title(name)
        axs[row][col].plot(vals,losses,linestyle='-',color='black',linewidth=2)
        if ylims is not None:
            axs[row][col].set_ylim(min(ylims),max(ylims))
        else:
            ylims_dict[name] = [min(losses),max(losses)]
        # Reset param
        config[name] = orig_val
    for n in range(len(params),shape[0]*shape[1]):
        row, col = (n-n % shape[0])//shape[0], n % shape[0]
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    # Scale ylims to suitable values if not provided beforehand
    if ylims is None:
        minlims = np.array([min(ylims_dict[name]) for name in params])
        minlims = minlims[~np.isinf(minlims) & ~np.isnan(minlims)]
        min_min = np.min(minlims)
        min_median = np.median(minlims)
        
        ymin = min_median - 1.1 * (min_median-min_min)
        #maxlims = np.array([max(ylims_dict[name]) for name in params])
        #maxlims = maxlims[~np.isinf(maxlims) & ~np.isnan(maxlims)]
        #min_max, max_max = np.min(maxlims), np.max(maxlims)
        ymax = min_median + 1.5 * (min_median-min_min)
        for n in range(len(params)):
            row, col = (n-n % shape[0])//shape[0], n % shape[0]
            axs[row][col].set_ylim(ymin,ymax)
    plt.show()
    
def plot_temperature(scenario,start,end,prior,posterior,MAP,data,std,save_path=None):
    years = list(range(start,end+1))
    T_fig = plt.figure('T_figure',figsize=(10,5))
    ax_T = T_fig.gca()
    ax_T.set_title(f'Temperature trend, scenario {scenario}')
    ax_T.set_xlabel('year')
    ax_T.set_ylabel('Temperature (°C)')
    plt.ylim(-1.5,4)
    
    ax_T.fill_between(data.year,data.data-std.data*1.96,data.data+std.data*1.96,color='red',alpha=0.5)
    
    prior = prior.temperature.loc[dict(timebounds=slice(start,end),scenario=scenario,layer=0)]
    prior_mean, prior_std = prior.mean(axis=1), prior.std(axis=1)
    ax_T.plot(years, prior_mean, linestyle='-', color='red', label='prior')
    ax_T.fill_between(years,prior_mean-prior_std*1.96,prior_mean+prior_std*1.96,color='red',alpha=0.3)
    
    posterior = posterior.temperature.loc[dict(timebounds=slice(start,end),scenario=scenario,layer=0)]
    posterior_mean, posterior_std = posterior.mean(axis=1), posterior.std(axis=1)
    ax_T.plot(years, posterior_mean, linestyle='-', color='blue', markersize=2, label='posterior mean')
    ax_T.fill_between(years,posterior_mean-posterior_std*1.96,posterior_mean+posterior_std*1.96,color='blue',alpha=0.3)
    
    MAP_temperature = MAP.temperature.loc[dict(timebounds=slice(start,end),scenario=scenario,layer=0)]
    ax_T.plot(years, MAP_temperature, linestyle='-', color='black', markersize=2, label='maximum posterior')
    
    ax_T.plot(data.year, data.data, linestyle='-', color='orange', label='Temperature observation')
    ax_T.plot(data.year, data.data-1.96*std.data,linestyle='--',color='orange')
    ax_T.plot(data.year, data.data+1.96*std.data,linestyle='--',color='orange')
    
    ax_T.legend(loc='upper left')
    if save_path is not None:
        T_fig.savefig(save_path)
    plt.show()
    
def plot_specie(start,end,prior,posterior,save_path=None):
    specie_fig = plt.figure('specie')
    ax_specie = specie_fig.gca()
    years = list(range(start,end+1))
    ax_specie.plot(years, prior[(start-1750):(end-1750+1)], linestyle='--', color='black', markersize=2, label='prior')
    ax_specie.plot(years, posterior[(start-1750):(end-1750+1)], linestyle='--', color='black', markersize=2, label='posterior')
    ax_specie.set_title('Specie concentration')
    ax_specie.set_xlabel('year')
    ax_specie.set_ylabel('concentration')
    if save_path is not None:
        specie_fig.savefig(save_path)
    plt.show()
    
def plot_constraints(MC_sampling,constraint_ranges,constraint_priors,constraint_targets,
                     constraint_posteriors=None,N=10000):
    colors = {"prior": "#207F6E", "posterior": "#EE696B", "target": "black"}
    
    # Initiate figure
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    
    ecs_axis = np.linspace(min(constraint_ranges['ecs']),max(constraint_ranges['ecs']),1000)
    prior = constraint_priors['ecs']
    target = constraint_targets['ecs']
    posterior = gaussian_kde(MC_sampling['ecs'][-N:].data,bw_method=lambda x: 0.1)
    ax[0, 0].plot(ecs_axis,prior(ecs_axis),color=colors["prior"],label="Prior")
    ax[0, 0].plot(ecs_axis,target(ecs_axis),color=colors["target"],label="Target")
    ax[0, 0].plot(ecs_axis,posterior(ecs_axis),color=colors["posterior"],label="Posterior")
    if constraint_posteriors is not None:
        ax[0, 0].plot(ecs_axis,constraint_posteriors['ecs'](ecs_axis),color='blue')
    
    ax[0, 0].set_xlim(0, 8)
    #ax[0, 0].set_ylim(0, 0.5)
    ax[0, 0].set_title("ECS")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("°C")
    
    tcr_axis = np.linspace(min(constraint_ranges['tcr']),max(constraint_ranges['tcr']),1000)
    target = constraint_targets['tcr']
    prior = constraint_priors['tcr']
    posterior = gaussian_kde(MC_sampling['ecs'][-N:].data)
    ax[0, 1].plot(tcr_axis,prior(tcr_axis),color=colors["prior"],label="Prior")
    ax[0, 1].plot(tcr_axis,target(tcr_axis),color=colors["target"],label="Target")
    ax[0, 1].plot(tcr_axis,posterior(tcr_axis),color=colors["posterior"],label="Posterior")
    ax[0, 1].set_xlim(0, 4)
    #ax[0, 1].set_ylim(0, 1.4)
    ax[0, 1].set_title("TCR")
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xlabel("°C")
    if constraint_posteriors is not None:
        ax[0, 1].plot(tcr_axis,constraint_posteriors['tcr'](tcr_axis),color='blue')

    past_anomaly_axis = np.linspace(min(constraint_ranges['T 1995-2014']),max(constraint_ranges['T 1995-2014']),1000)
    prior = constraint_priors['T 1995-2014']
    target = constraint_targets['T 1995-2014']
    posterior = gaussian_kde(MC_sampling['T 1995-2014'][-N:].data)
    ax[0, 2].plot(past_anomaly_axis,prior(past_anomaly_axis),color=colors["prior"],label="Prior")
    ax[0, 2].plot(past_anomaly_axis,target(past_anomaly_axis),color=colors["target"],label="Target")
    ax[0, 2].plot(past_anomaly_axis,posterior(past_anomaly_axis),color=colors["posterior"],label="Posterior")
    ax[0, 2].set_xlim(0.5, 1.3)
    #ax[0, 2].set_ylim(0, 5)
    ax[0, 2].set_title("Past temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 1995-2014 minus 1850-1900")
    if constraint_posteriors is not None:
        ax[0, 2].plot(past_anomaly_axis,constraint_posteriors['T 1995-2014'](past_anomaly_axis),color='blue')

    ari_axis = np.linspace(min(constraint_ranges['ari']),max(constraint_ranges['ari']),1000)
    prior = constraint_priors['ari']
    target = constraint_targets['ari']
    posterior = gaussian_kde(MC_sampling['ari'][-N:].data)
    ax[1, 0].plot(ari_axis,prior(ari_axis),color=colors["prior"],label="Prior")
    ax[1, 0].plot(ari_axis,target(ari_axis),color=colors["target"],label="Target")
    ax[1, 0].plot(ari_axis,posterior(ari_axis),color=colors["posterior"],label="Posterior")
    ax[1, 0].set_xlim(-1.0, 0.3)
    #ax[1, 0].set_ylim(0, 3)
    ax[1, 0].set_title("Aerosol ERFari")
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    if constraint_posteriors is not None:
        ax[1, 0].plot(ari_axis,constraint_posteriors['ari'](ari_axis),color='blue')
    
    aci_axis = np.linspace(min(constraint_ranges['aci']),max(constraint_ranges['aci']),1000)
    prior = constraint_priors['aci']
    target = constraint_targets['aci']
    posterior = gaussian_kde(MC_sampling['aci'][-N:].data)
    ax[1, 1].plot(aci_axis,prior(aci_axis),color=colors["prior"],label="Prior")
    ax[1, 1].plot(aci_axis,target(aci_axis),color=colors["target"],label="Target")
    ax[1, 1].plot(aci_axis,posterior(aci_axis),color=colors["posterior"],label="Posterior MC")
    ax[1, 1].set_xlim(-2.25, 0.25)
    #ax[1, 1].set_ylim(0, 1.6)
    ax[1, 1].set_title("Aerosol ERFaci")
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    if constraint_posteriors is not None:
        ax[1, 1].plot(aci_axis,constraint_posteriors['aci'](aci_axis),color='blue',label='Posterior orig')
    ax[1, 1].legend(frameon=False)

    aer_axis = np.linspace(min(constraint_ranges['aer']),max(constraint_ranges['aer']),1000)
    prior = constraint_priors['aer']
    target = constraint_targets['aer']
    posterior = gaussian_kde(MC_sampling['aer'][-N:].data)
    ax[1, 2].plot(aer_axis,target(aer_axis),color=colors["target"],label="Target")
    ax[1, 2].plot(aer_axis,prior(aer_axis),color=colors["prior"],label="Prior")
    ax[1, 2].plot(aer_axis,posterior(aer_axis),color=colors["posterior"],label="Posterior")
    ax[1, 2].set_xlim(-3, 0)
    #ax[1, 2].set_ylim(0, 1.6)
    ax[1, 2].set_title("Aerosol ERF")
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    if constraint_posteriors is not None:
        ax[1, 2].plot(aer_axis,constraint_posteriors['aer'](aer_axis),color='blue')

    co2_axis = np.linspace(min(constraint_ranges['CO2']),max(constraint_ranges['CO2']),1000)
    prior = constraint_priors['CO2']
    target = constraint_targets['CO2']
    posterior = gaussian_kde(MC_sampling['CO2'][-N:].data)
    ax[2, 0].plot(co2_axis,prior(co2_axis),color=colors["prior"],label="Prior")
    ax[2, 0].plot(co2_axis,target(co2_axis),color=colors["target"],label="Target")
    ax[2, 0].plot(co2_axis,posterior(co2_axis),color=colors["posterior"],label="Posterior")
    ax[2, 0].set_xlim(394, 402)
    #ax[2, 0].set_ylim(0, 1.2)
    ax[2, 0].set_title("CO$_2$ concentration")
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xlabel("ppm, 2014")
    if constraint_posteriors is not None:
        ax[2, 0].plot(co2_axis,constraint_posteriors['CO2'](co2_axis),color='blue')

    ohc_axis = np.linspace(min(constraint_ranges['ohc']),max(constraint_ranges['ohc']),1000)
    prior = constraint_priors['ohc']
    target = constraint_targets['ohc']
    posterior = gaussian_kde(MC_sampling['ohc'][-N:].data)
    ax[2, 1].plot(ohc_axis,prior(ohc_axis),color=colors["prior"],label="Prior")
    ax[2, 1].plot(ohc_axis,target(ohc_axis),color=colors["target"],label="Target")
    ax[2, 1].plot(ohc_axis,posterior(ohc_axis),color=colors["posterior"],label="Posterior")
    ax[2, 1].set_xlim(0, 800)
    #ax[2, 1].set_ylim(0, 0.006)
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2018 minus 1971")
    if constraint_posteriors is not None:
        ax[2, 1].plot(ohc_axis,constraint_posteriors['ohc'](ohc_axis),color='blue')

    future_anomaly_axis = np.linspace(min(constraint_ranges['T 2081-2100']),max(constraint_ranges['T 2081-2100']),1000)
    prior = constraint_priors['T 2081-2100']
    target = constraint_targets['T 2081-2100']
    posterior = gaussian_kde(MC_sampling['T 2081-2100'][-N:].data)
    ax[2, 2].plot(future_anomaly_axis,prior(future_anomaly_axis),color=colors["prior"],label="Prior")
    ax[2, 2].plot(future_anomaly_axis,target(future_anomaly_axis),color=colors["target"],label="Target")
    ax[2, 2].plot(future_anomaly_axis,posterior(future_anomaly_axis),color=colors["posterior"],label="Posterior")
    ax[2, 2].set_xlim(0.8, 3.2)
    #ax[2, 2].set_ylim(0, 1.1)
    ax[2, 2].set_title("Future temperature anomaly")
    ax[2, 2].set_yticklabels([])
    ax[2, 2].set_xlabel("°C, 2081-2100 minus 1995-2014, ssp245")
    if constraint_posteriors is not None:
        ax[2, 2].plot(future_anomaly_axis,constraint_posteriors['T 2081-2100'](future_anomaly_axis),color='blue')

    fig.tight_layout()
    plt.show()
    