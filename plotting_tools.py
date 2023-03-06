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
        axs[row][col].hist(data,bins=20,density=True)
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
            axs[row][col].hist(pos_configs[name],bins=30,density=True)
    for n in range(len(col_names),7*7):
        row, col = (n-n%7)//7, n%7
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    #fig.savefig('plots/distributions')
    plt.show()
    
def plot_temperature(scenario,start,end,prior,posterior,data,std,save_path=None):
    years = list(range(start,end+1))
    T_fig = plt.figure('T',figsize=(10,5))
    ax_T = T_fig.gca()
    ax_T.plot(years, prior[(start-1750):(end-1750+1)], linestyle='--', color='black', markersize=2, label='prior')
    ax_T.plot(years, posterior[(start-1750):(end-1750+1)], linestyle='-', color='black', markersize=2, label='posterior')
    #ax_T.errorbar(range(1850,2021), y=data, yerr=std, fmt='.',capsize=5,markersize=1,color='orange')
    #ax_T.plot(range(1850,2021), data, 'r.', markersize=3)
    ax_T.plot(range(1850,2021), data, 'r-', label='Global T trend')
    ax_T.set_title(f'Temperature trend, scenario {scenario}')
    ax_T.set_xlabel('year')
    ax_T.set_ylabel('Temperature (°C)')
    plt.ylim(-1.5,3)
    ax_T.legend()
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
    