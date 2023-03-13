#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:12:03 2023

@author: nurmelaj
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm, uniform

def plot_distributions(prior_distributions,prior_configs,pos_configs,exclude=[]):
    fig, axs = plt.subplots(7,7,figsize=(25,20))
    #fig.canvas.set_window_title('Distributions')
    fig.tight_layout(pad=5)
    plt.ticklabel_format(useOffset=False)
    col_names = prior_configs.columns
    for n in range(len(col_names)):
        name = col_names[n]
        row, col = (n-n%7)//7, n%7
        data = prior_configs[name].values
        axs[row][col].hist(data,bins=40,density=True)
        axs[row][col].set_title(name)
        xmin, xmax = np.min(data), np.max(data)
        x = np.linspace(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin),100)
        if prior_distributions[name]['distribution'] == 'gamma':
            axs[row][col].plot(x,gamma.pdf(x,*prior_distributions[name]['params']),linestyle='--',color='black')
        elif prior_distributions[name]['distribution'] == 'norm':
            axs[row][col].plot(x,norm.pdf(x,*prior_distributions[name]['params']),linestyle='--',color='black')
        else:
            axs[row][col].plot(x,uniform.pdf(x,*prior_distributions[name]['params']),linestyle='--',color='black')
        if name not in exclude:
            axs[row][col].hist(pos_configs[name],bins=40,density=True)
    for n in range(len(col_names),7*7):
        row, col = (n-n%7)//7, n%7
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    #fig.savefig('plots/distributions')
    plt.show()
    
def plot_temperature(scenario,start,end,prior,posterior,opt,data,std,save_path=None):
    years = list(range(start,end+1))
    T_fig = plt.figure('T',figsize=(10,5))
    ax_T = T_fig.gca()
    ax_T.set_title(f'Temperature trend, scenario {scenario}')
    ax_T.set_xlabel('year')
    ax_T.set_ylabel('Temperature (°C)')
    plt.ylim(-1.5,4)
    
    ax_T.plot(data.year, data.data, linestyle='-', color='orange', label='Temperature observation')
    #ax_T.fill_between(data.year,data.data-std.data*1.96,data.data+std.data*1.96,color='red',alpha=0.5)
    
    prior = prior.temperature.loc[dict(timebounds=slice(start,end),scenario=scenario,layer=0)]
    prior_mean, prior_std = prior.mean(axis=1), prior.std(axis=1)
    ax_T.plot(years, prior_mean, linestyle='-', color='red', label='prior')
    #ax_T.fill_between(years,prior_mean-prior_std*1.96,prior_mean+prior_std*1.96,color='green',alpha=0.3)
    
    #posterior = posterior.temperature.loc[dict(timebounds=slice(start,end),scenario=scenario,layer=0)]
    #posterior_mean, posterior_std = posterior.mean(axis=1), posterior.std(axis=1)
    #ax_T.plot(years, posterior_mean, linestyle='-', color='blue', markersize=2, label='posterior')
    #ax_T.fill_between(years,posterior_mean-posterior_std*1.96,posterior_mean+posterior_std*1.96,color='yellow',alpha=0.3)
    
    optimal = opt.temperature.loc[dict(timebounds=slice(start,end),scenario=scenario,layer=0)]
    ax_T.plot(years, optimal, linestyle='-', color='black', markersize=2, label='MAP')
    
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
    