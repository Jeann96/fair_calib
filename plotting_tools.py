#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2023

@author: nurmelaj
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import concat
from scipy.stats import gaussian_kde,norm
# Import data reading tools
from fair_tools import read_forcing_data,read_gmst_temperature,read_hadcrut_temperature,compute_trends,constraint_targets
from fair_tools import runFaIR,compute_data_loss,compute_prior_loss,compute_constrained_loss,compute_constraints,resample_constraint_posteriors,get_param_ranges,get_log_prior
# dotenv parameters
from fair_tools import cwd,cal_v,fair_v,constraint_set,constraint_ranges,get_log_constraint_target

def plot_distributions(prior_configs,pos_configs,alpha=0.5,prior_color=None,pos_color=None,
                       title=None,savefig=False):
    if prior_color is None or pos_color is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        prior_color = default_colors[0] if prior_color is None else prior_color
        pos_color = default_colors[1] if pos_color is None else pos_color
    fig, axs = plt.subplots(7,7,figsize=(25,20))
    fig.tight_layout(pad=6)
    plt.ticklabel_format(useOffset=False)
    
    param_ranges = get_param_ranges()
    prior_names, pos_names = prior_configs.columns, pos_configs.columns
    for n in range(len(prior_names)):
        name = prior_names[n]
        row, col = (n-n%7)//7, n%7
        axs[row][col].hist(prior_configs[name].values,density=True,alpha=alpha,range=param_ranges[name],
                           bins=int(np.ceil(1.5*np.sqrt(len(prior_configs)))),color=prior_color)
        if name in pos_names:
            axs[row][col].hist(pos_configs[name].values,density=True,alpha=alpha,range=param_ranges[name],
                               bins=int(np.ceil(1.5*np.sqrt(len(pos_configs)))),color=pos_color)
        axs[row][col].set_title(name)
    
    extra_params = [param for param in pos_names if param not in prior_names]
    if len(extra_params) != 0:
        for n in range(len(prior_names),len(prior_names)+len(extra_params)):
            name = extra_params[n-len(prior_names)]
            row, col = (n-n%7)//7, n%7
            axs[row][col].hist(pos_configs[name].values,density=True,alpha=alpha,range=param_ranges[name],
                               bins=int(np.ceil(1.5*np.sqrt(len(pos_configs)))),color=pos_color)
            axs[row][col].set_title(name)
    
    
    for n in range(n+1,7*7):
        row, col = (n-n%7)//7, n%7
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    if title is not None:
        plt.suptitle(title,fontsize=20)
    if savefig:
        save_name = title.lower().replace(' ', '_')
        plt.savefig(f"figures/distributions/{save_name}")
    else:
        plt.show()
        
def sensitivity_test(config,included,param_ranges,ylims=None,stochastic_run=True,use_constraints=True,
                     data_loss_method='wss',N=500,scenario='ssp245',savefig=True):
    # Read data
    gmst = read_gmst_temperature()
    gmst = gmst.rename({'year':'timebounds'})
    T = read_hadcrut_temperature()
    T = T.rename({'year':'timebounds'})
    if data_loss_method == 'wss':
        obs, var = gmst, T.std(dim='realization')**2
    elif data_loss_method == 'trend':
        trends = compute_trends()
        obs, var = np.mean(trends), np.std(trends)**2

    solar_forcing, volcanic_forcing, emissions = read_forcing_data(scenario,N,1750,2101)
    configs = concat([config]*N, ignore_index=True, axis='rows')
    log_prior = get_log_prior(included)
    if use_constraints:
        log_constraint_target = get_log_constraint_target()
    
    shape = layout_shape(len(included))
    fig, axs = plt.subplots(shape[1],shape[0],figsize=(35,20))
    fig.tight_layout(pad=6)
    plt.ticklabel_format(useOffset=False)
    if ylims is None:
        ylims_dict = {}
    for n in range(len(included)):
        param = included[n]
        print(param)
        # List of different values for a parameter
        if param == 'gamma':
            param_min, param_max = 0.25, max(param_ranges[param])
        else:
            param_min, param_max = min(param_ranges[param]),max(param_ranges[param])
        configs[param] = np.linspace(param_min,param_max,N)
        fair = runFaIR(solar_forcing,volcanic_forcing,emissions,configs,scenario,
                       start=1750,end=2101,stochastic_run=stochastic_run)
        anomaly = fair.temperature.sel(timebounds=slice(1851,2021),scenario=scenario,layer=0) -\
                  fair.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds')
        losses = compute_data_loss(anomaly,obs,var,method=data_loss_method)
        losses += compute_prior_loss(log_prior,configs[included].to_numpy())
        if use_constraints:
            constraints = compute_constraints(fair)
            losses += compute_constrained_loss(constraints,log_constraint_target)
            
        row, col = (n - n % shape[0])//shape[0], n % shape[0]
        axs[row][col].plot(configs[param].values,losses,'k-')
        if ylims is not None:
            axs[row][col].set_ylim(min(ylims),max(ylims))
        else:
            ylims_dict[param] = [min(losses),max(losses)]
        #axs[row][col].set_xlabel(param)
        axs[row][col].set_title(param,fontsize=12)
        
        # Reset param
        configs[param] = np.repeat(float(config[param]),N)
    
    for n in range(n+1,np.prod(shape)):
        row, col =  (n - n % shape[0])//shape[0], n % shape[0]
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    
    if ylims is None:
        minlims = np.array([min(ylims_dict[param]) for param in included])
        minlims = minlims[~np.isinf(minlims) & ~np.isnan(minlims)]
        min_min = np.min(minlims)
        min_median = np.median(minlims)
        ymin = min_median - 1.1 * (min_median-min_min)
        ymax = min_median + 1.5 * (min_median-min_min)
        for n in range(len(included)):
            row, col = (n - n % shape[0])//shape[0], n % shape[0]
            axs[row][col].set_ylim(ymin,ymax)
        
    #plt.suptitle(f'Sentivity test, {data_loss_method} loss function',fontsize=24)
    if savefig:
        plt.savefig(f'figures/sensitivity/loss_sensitivity_{data_loss_method}')
    else:
        plt.show()
    
def layout_shape(N,threshold=3):
    '''
    Find suitable layout for several images.
    '''
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
    N = len(chain)
    shape = layout_shape(len(params))
    fig, axs = plt.subplots(shape[1],shape[0],figsize=(30,25))
    fig.tight_layout(pad=5)
    plt.ticklabel_format(useOffset=False)
    for n in range(len(params)):
        name = params[n]
        row, col = (n-n % shape[0])//shape[0], n % shape[0]
        axs[row][col].plot(chain[:,n],linestyle='-',color='black',linewidth=0.5)
        ticks = [0,N//4,N//2,(3*N)//4,N]
        axs[row][col].set_xticks(ticks=ticks,labels=[f'{int(tick)}' for tick in ticks])
        axs[row][col].set_title(name)
    for n in range(len(params),shape[0]*shape[1]):
        row, col = (n-n % shape[0])//shape[0], n % shape[0]
        axs[row][col].set_xticks(ticks=[],labels=[])
        axs[row][col].set_yticks(ticks=[],labels=[])
    plt.show()
    
def plot_temperature(scenario,start,end,prior,posterior=None,MAP=None,obs=None,obs_std=None,
                     plot_trend=False,savefig=True):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    years = list(range(start,end+1))
    T_fig = plt.figure('T_figure',figsize=(10,5))
    ax_T = T_fig.gca()
    ax_T.set_title(f'Temperature trend, scenario {scenario}')
    ax_T.set_xlabel('year')
    ax_T.set_ylabel('Temperature (°C)')
    plt.ylim(-1.5,4)
    
    prior_T = prior.temperature.sel(timebounds=slice(start,end),scenario=scenario,layer=0) \
            - prior.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
    prior_mean = prior_T.mean(axis=1).data.squeeze()
    ax_T.plot(years, prior_mean, linestyle='-', color=colors[0], label='calib')
    if obs is None and obs_std is not None:
        prior_slice = prior.temperature.loc[dict(timebounds=obs_std.year.data,scenario=scenario,layer=0)].data.squeeze()
        ax_T.fill_between(obs_std.year.data,prior_slice-obs_std.data*1.96,prior_slice+obs_std.data*1.96,color=colors[2],alpha=0.4)
    else:
        prior_std = prior_T.std(axis=1)
        ax_T.fill_between(years,prior_mean-prior_std.data*1.96,prior_mean+prior_std.data*1.96,color=colors[0],alpha=0.4)
    
    if posterior is not None:
        posterior = posterior.temperature.sel(timebounds=slice(start,end),scenario=scenario,layer=0) \
                  - posterior.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
        posterior_mean = posterior.mean(axis=1).data.squeeze()
        posterior_std = posterior.std(axis=1).data.squeeze()
        ax_T.plot(years, posterior_mean, linestyle='-', color=colors[1], markersize=2, label='posterior')
        ax_T.fill_between(years,posterior_mean-posterior_std*1.96,posterior_mean+posterior_std*1.96,color=colors[1],alpha=0.4)
    
    if MAP is not None:
        MAP_temperature = MAP.temperature.sel(timebounds=slice(start,end),scenario=scenario,layer=0) \
                        - MAP.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
        ax_T.plot(years, MAP_temperature.data.squeeze(), linestyle='-', color='black', markersize=2, label='MAP')
    
    if obs is not None:
        ax_T.plot(obs.year, obs.data, linestyle='-', color=colors[2], label='observation')
        if obs_std is not None:
            ax_T.fill_between(obs.year,obs.data-1.96*obs_std.data,obs.data+1.96*obs_std.data,color=colors[2],alpha=0.4)
        if plot_trend:
            x = np.arange(1900,2021,1)
            y = obs.sel(year=slice(1900,2020)).data
            params = np.polyfit(x,y,1)
            ax_T.plot(x,params[0] * x + params[1], label='trend line 1900-2020', color='black')
    
    ax_T.legend(loc='upper left')
    if savefig:
        plt.savefig(f'figures/trends/temperature_trend_{min(years)}-{max(years)}')
        #plt.savefig(f'figures/stochastic_vs_deterministic_{min(years)}-{max(years)}')
    else:
        return T_fig
    
def plot_specie(start,end,prior,posterior,savefig=True):
    '''
    Unfinished
    '''
    specie_fig = plt.figure('specie')
    ax_specie = specie_fig.gca()
    years = list(range(start,end+1))
    ax_specie.plot(years, prior[(start-1750):(end-1750+1)], linestyle='--', color='black', markersize=2, label='prior')
    ax_specie.plot(years, posterior[(start-1750):(end-1750+1)], linestyle='--', color='black', markersize=2, label='posterior')
    ax_specie.set_title('Specie concentration')
    ax_specie.set_xlabel('year')
    ax_specie.set_ylabel('concentration')
    if savefig:
        plt.savefig('figures/trends/specie_concentration')
    else:
        plt.show()
        
def plot_trends(T,start=1900,end=2020,savefig=True):
    trends = compute_trends(start=start,end=end)
    trend_fig = plt.figure('trend_hist',figsize=(10,5))
    mean, std = np.mean(trends), np.std(trends)
    ax = trend_fig.gca()
    ax.hist(trends,bins=int(np.ceil(np.sqrt(1.1*len(trends)))),density=True,edgecolor='black',label='histogram')
    x = np.linspace(min(trends),max(trends),500)
    ax.plot(x,norm.pdf(x, mean, std), 'k-', label='normal density')
    ax.set_title(f'Trends {start}-{end}')
    ax.set_xlabel('slope')
    ax.legend()
    if savefig:
        plt.savefig(f'figures/trends/trend_histogram_{start}-{end}')
    else:
        plt.show()
    
def plot_constraints(MC_sampling,N=1000,thinning=1,plot_histo=True,alpha=0.6,colors=None,savefig=True):
    if colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {'prior': '#207F6E', 'posterior_calib': default_colors[0], 
                  'posterior_mc': default_colors[1], 'target': 'black'}
    required_samples = N * thinning
    samples = MC_sampling.dimensions['sample'].size - int(MC_sampling['warmup'][:].data)
    if required_samples > samples:
        raise ValueError(f'Required samples {required_samples} with thinning {thinning} is too large')
    # Constraint ranges (min,max) and target densities
    ranges = constraint_ranges()
    targets = constraint_targets()
    # Prior data
    temp = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy")
    prior_ecs_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy")
    prior_tcr_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy")
    prior_ari_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ari_2005-2014_mean.npy")
    prior_aci_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_aci_2005-2014_mean.npy")
    prior_aer_samples = prior_aci_samples + prior_ari_samples
    prior_co2_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/concentration_co2_2014.npy")
    prior_ohc_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ocean_heat_content_2018_minus_1971.npy") / 1e21
    prior_past_anomaly_samples = np.average(temp[145:166,:],axis=0) - np.average(temp[:52,:],axis=0)
    prior_future_anomaly_samples = np.average(temp[231:252,:],axis=0) - np.average(temp[145:166,:],axis=0)
    
    print('Resampling...')
    calib_samples = resample_constraint_posteriors()
    print('...done')
    calib_ecs_samples = calib_samples['ecs'].to_numpy()
    calib_tcr_samples = calib_samples['tcr'].to_numpy()
    calib_ari_samples = calib_samples['ari'].to_numpy()
    calib_aci_samples = calib_samples['aci'].to_numpy()
    calib_aer_samples = calib_samples['aer'].to_numpy()
    calib_co2_samples = calib_samples['CO2'].to_numpy()
    calib_ohc_samples = calib_samples['ohc'].to_numpy()
    calib_past_anomaly_samples = calib_samples['T 1995-2014'].to_numpy()
    calib_future_anomaly_samples = calib_samples['T 2081-2100'].to_numpy()
    
    mc_ecs_samples = MC_sampling['ecs'][-required_samples::thinning].data
    mc_tcr_samples = MC_sampling['tcr'][-required_samples::thinning].data
    mc_ari_samples = MC_sampling['ari'][-required_samples::thinning].data
    mc_aci_samples = MC_sampling['aci'][-required_samples::thinning].data
    mc_aer_samples = MC_sampling['aer'][-required_samples::thinning].data
    mc_co2_samples = MC_sampling['CO2'][-required_samples::thinning].data
    mc_ohc_samples = MC_sampling['ohc'][-required_samples::thinning].data
    mc_past_anomaly_samples = MC_sampling['T 1995-2014'][-required_samples::thinning].data
    mc_future_anomaly_samples = MC_sampling['T 2081-2100'][-required_samples::thinning].data
    
    # Initiate figure
    print('Plotting constraints...')
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    
    print('ECS')
    ecs_axis = np.linspace(min(ranges['ecs']),max(ranges['ecs']),1000)
    target = targets['ecs']
    posterior_mc = gaussian_kde(mc_ecs_samples)
    kd_prior_ecs = gaussian_kde(prior_ecs_samples)
    kd_calib_ecs = gaussian_kde(calib_ecs_samples)
    if plot_histo:
        #ax[0, 0].hist(prior_ecs_samples,bins=int(np.floor(np.sqrt(len(prior_ecs_samples)))),density=True,color=colors['prior'])
        ax[0, 0].hist(mc_ecs_samples,bins=int(np.floor(np.sqrt(len(mc_ecs_samples)))),density=True,
                      color=colors['posterior_mc'],alpha=alpha,label='pos MC')
        ax[0, 0].hist(calib_ecs_samples,bins=int(np.floor(np.sqrt(len(calib_ecs_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[0, 0].plot(ecs_axis,kd_prior_ecs(ecs_axis),color=colors["prior"],label="prior")
    ax[0, 0].plot(ecs_axis,target(ecs_axis),color=colors["target"],label="target")
    ax[0, 0].plot(ecs_axis,kd_calib_ecs(ecs_axis),color=colors['posterior_calib'])
    ax[0, 0].plot(ecs_axis,posterior_mc(ecs_axis),color=colors['posterior_mc'])
    ax[0, 0].set_xlim(1, 7)
    ax[0, 0].set_title("ECS")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("°C")
    
    print('TCR')
    tcr_axis = np.linspace(min(ranges['tcr']),max(ranges['tcr']),1000)
    target = targets['tcr']
    kd_prior_tcr = gaussian_kde(prior_tcr_samples)
    kd_calib_tcr = gaussian_kde(calib_tcr_samples)
    posterior_mc = gaussian_kde(mc_tcr_samples)
    if plot_histo:
        #ax[0, 1].hist(prior_tcr_samples,bins=int(np.floor(np.sqrt(len(prior_tcr_samples)))),density=True,color=colors['prior'])
        ax[0, 1].hist(mc_tcr_samples,bins=int(np.floor(np.sqrt(len(mc_tcr_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[0, 1].hist(calib_tcr_samples,bins=int(np.floor(np.sqrt(len(calib_tcr_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[0, 1].plot(tcr_axis,kd_prior_tcr(tcr_axis),color=colors["prior"],label="prior")
    ax[0, 1].plot(tcr_axis,target(tcr_axis),color=colors["target"],label="target")
    ax[0, 1].plot(tcr_axis,kd_calib_tcr(tcr_axis),color=colors['posterior_calib'])
    ax[0, 1].plot(tcr_axis,posterior_mc(tcr_axis),color=colors['posterior_mc'])
    ax[0, 1].set_xlim(0.5, 3.5)
    ax[0, 1].set_title("TCR")
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xlabel("°C")
    
    print('Temperature 1995-2014 minus 1850-1900')
    past_anomaly_axis = np.linspace(min(ranges['T 1995-2014']),max(ranges['T 1995-2014']),1000)
    kd_prior_past_anomaly = gaussian_kde(prior_past_anomaly_samples)
    kd_calib_past_anomaly = gaussian_kde(calib_past_anomaly_samples)
    target = targets['T 1995-2014']
    posterior_mc = gaussian_kde(MC_sampling['T 1995-2014'][-required_samples::thinning].data)
    if plot_histo:
        # ax[0, 2].hist(prior_past_anomaly_samples,bins=int(np.floor(np.sqrt(len(prior_past_anomaly_samples)))),density=True,color=colors['prior'])
        ax[0, 2].hist(mc_past_anomaly_samples,bins=int(np.floor(np.sqrt(len(mc_past_anomaly_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[0, 2].hist(calib_past_anomaly_samples,bins=int(np.floor(np.sqrt(len(calib_past_anomaly_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[0, 2].plot(past_anomaly_axis,kd_prior_past_anomaly(past_anomaly_axis),color=colors["prior"],label="Prior")
    ax[0, 2].plot(past_anomaly_axis,target(past_anomaly_axis),color=colors["target"],label="target")
    ax[0, 2].plot(past_anomaly_axis,kd_calib_past_anomaly(past_anomaly_axis),color=colors['posterior_calib'])
    ax[0, 2].plot(past_anomaly_axis,posterior_mc(past_anomaly_axis),color=colors['posterior_mc'])
    ax[0, 2].set_xlim(0.5, 1.2)
    #ax[0, 2].set_ylim(0, 5)
    ax[0, 2].set_title("Past temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 1995-2014 minus 1850-1900")
        
    print('Aerosol ERFari')
    ari_axis = np.linspace(min(ranges['ari']),max(ranges['ari']),1000)
    kd_prior_ari = gaussian_kde(prior_ari_samples)
    kd_calib_ari = gaussian_kde(calib_ari_samples)
    target = targets['ari']
    posterior_mc = gaussian_kde(MC_sampling['ari'][-required_samples::thinning].data)
    if plot_histo:
        # ax[1, 0].hist(prior_ari_samples,bins=int(np.floor(np.sqrt(len(prior_ari_samples)))),density=True,color=colors['prior'])
        ax[1, 0].hist(mc_ari_samples,bins=int(np.floor(np.sqrt(len(mc_ari_samples)))),density=True,
                     color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[1, 0].hist(calib_ari_samples,bins=int(np.floor(np.sqrt(len(calib_ari_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[1, 0].plot(ari_axis,kd_prior_ari(ari_axis),color=colors["prior"],label="prior")
    ax[1, 0].plot(ari_axis,target(ari_axis),color=colors["target"],label="target")
    ax[1, 0].plot(ari_axis,kd_calib_ari(ari_axis),color=colors['posterior_calib'])
    ax[1, 0].plot(ari_axis,posterior_mc(ari_axis),color=colors['posterior_mc'])
    ax[1, 0].set_xlim(-1.0, 0.4)
    #ax[1, 0].set_ylim(0, 3)
    ax[1, 0].set_title("Aerosol ERFari")
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    
    print('Aerosol ERFaci')
    aci_axis = np.linspace(min(ranges['aci']),max(ranges['aci']),1000)
    kd_prior_aci = gaussian_kde(prior_aci_samples)
    kd_calib_aci = gaussian_kde(calib_aci_samples)
    target = targets['aci']
    posterior_mc = gaussian_kde(MC_sampling['aci'][-required_samples::thinning].data)
    if plot_histo:
        # ax[1, 1].hist(prior_aci_samples,bins=int(np.floor(np.sqrt(len(prior_aci_samples)))),density=True,color=colors['prior'])
        ax[1, 1].hist(mc_aci_samples,bins=int(np.floor(np.sqrt(len(mc_aci_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[1, 1].hist(calib_aci_samples,bins=int(np.floor(np.sqrt(len(calib_aci_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[1, 1].plot(aci_axis,kd_prior_aci(aci_axis),color=colors['prior'],label='prior')
    ax[1, 1].plot(aci_axis,target(aci_axis),color=colors["target"],label="target")
    ax[1, 1].plot(aci_axis,kd_calib_aci(aci_axis),color=colors['posterior_calib'])
    ax[1, 1].plot(aci_axis,posterior_mc(aci_axis),color=colors['posterior_mc'])
    ax[1, 1].set_xlim(-2.25, 0.1)
    #ax[1, 1].set_ylim(0, 1.6)
    ax[1, 1].set_title("Aerosol ERFaci")
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    
    # Set legend
    ax[1, 1].legend(frameon=False)
    
    print('Aerosol ERF')
    aer_axis = np.linspace(min(ranges['aer']),max(ranges['aer']),1000)
    kd_prior_aer = gaussian_kde(prior_aer_samples)
    kd_calib_aer = gaussian_kde(calib_aer_samples)
    target = targets['aer']
    posterior_mc = gaussian_kde(MC_sampling['aer'][-required_samples::thinning].data)
    if plot_histo:
        # ax[1, 2].hist(prior_aer_samples,bins=int(np.floor(np.sqrt(len(prior_aer_samples)))),density=True,color=colors['prior'])
        ax[1, 2].hist(mc_aer_samples,bins=int(np.floor(np.sqrt(len(mc_aer_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[1, 2].hist(calib_aer_samples,bins=int(np.floor(np.sqrt(len(calib_aer_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[1, 2].plot(aer_axis,target(aer_axis),color=colors["target"],label="target")
    ax[1, 2].plot(aer_axis,kd_prior_aer(aer_axis),color=colors["prior"],label="prior")
    ax[1, 2].plot(aer_axis,kd_calib_aer(aer_axis),color=colors["posterior_calib"])
    ax[1, 2].plot(aer_axis,posterior_mc(aer_axis),color=colors['posterior_mc'])
    ax[1, 2].set_xlim(-3, 0)
    #ax[1, 2].set_ylim(0, 1.6)
    ax[1, 2].set_title("Aerosol ERF")
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    print('CO2 concentration')
    co2_axis = np.linspace(min(ranges['CO2']),max(ranges['CO2']),1000)
    kd_prior_co2 = gaussian_kde(prior_co2_samples)
    kd_calib_co2 = gaussian_kde(calib_co2_samples)
    target = targets['CO2']
    posterior_mc = gaussian_kde(MC_sampling['CO2'][-required_samples::thinning].data)
    if plot_histo:
        # ax[2, 0].hist(prior_co2_samples,bins=int(np.floor(np.sqrt(len(prior_co2_samples)))),density=True,color=colors['prior'])
        ax[2, 0].hist(mc_co2_samples,bins=int(np.floor(np.sqrt(len(mc_co2_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[2, 0].hist(calib_co2_samples,bins=int(np.floor(np.sqrt(len(calib_co2_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[2, 0].plot(co2_axis,kd_prior_co2(co2_axis),color=colors["prior"],label="prior")
    ax[2, 0].plot(co2_axis,target(co2_axis),color=colors["target"],label="target")
    ax[2, 0].plot(co2_axis,kd_calib_co2(co2_axis),color=colors["posterior_calib"])
    ax[2, 0].plot(co2_axis,posterior_mc(co2_axis),color=colors['posterior_mc'])
    ax[2, 0].set_xlim(396, 399)
    #ax[2, 0].set_ylim(0, 1.2)
    ax[2, 0].set_title("CO$_2$ concentration")
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xlabel("ppm, 2014")
    
    print('OHC change')
    ohc_axis = np.linspace(min(ranges['ohc']),max(ranges['ohc']),1000)
    kd_prior_ohc = gaussian_kde(prior_ohc_samples)
    kd_calib_ohc = gaussian_kde(calib_ohc_samples)
    target = targets['ohc']
    posterior_mc = gaussian_kde(MC_sampling['ohc'][-required_samples::thinning].data)
    if plot_histo:
        # ax[2, 1].hist(prior_ohc_samples,bins=int(np.floor(np.sqrt(len(prior_ohc_samples)))),density=True,color=colors['prior'])
        ax[2, 1].hist(mc_ohc_samples,bins=int(np.floor(np.sqrt(len(mc_ohc_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[2, 1].hist(calib_ohc_samples,bins=int(np.floor(np.sqrt(len(calib_ohc_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[2, 1].plot(ohc_axis,kd_prior_ohc(ohc_axis),color=colors["prior"],label="prior")
    ax[2, 1].plot(ohc_axis,target(ohc_axis),color=colors["target"],label="target")
    ax[2, 1].plot(ohc_axis,kd_calib_ohc(ohc_axis),color=colors["posterior_calib"])
    ax[2, 1].plot(ohc_axis,posterior_mc(ohc_axis),color=colors['posterior_mc'])
    ax[2, 1].set_xlim(0, 800)
    #ax[2, 1].set_ylim(0, 0.006)
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2018 minus 1971")

    print('Temperature 2081-2100 minus 1995-2014')    
    future_anomaly_axis = np.linspace(min(ranges['T 2081-2100']),max(ranges['T 2081-2100']),1000)
    kd_prior_future_anomaly = gaussian_kde(prior_future_anomaly_samples)
    kd_calib_future_anomaly = gaussian_kde(calib_future_anomaly_samples)
    target = targets['T 2081-2100']
    posterior_mc = gaussian_kde(MC_sampling['T 2081-2100'][-required_samples::thinning].data)
    if plot_histo:
        # ax[2, 2].hist(prior_future_anomaly_samples,bins=int(np.floor(np.sqrt(len(prior_future_anomaly_samples)))),density=True,color=colors['prior'])
        ax[2, 2].hist(mc_future_anomaly_samples,bins=int(np.floor(np.sqrt(len(mc_future_anomaly_samples)))),density=True,
                      color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        ax[2, 2].hist(calib_future_anomaly_samples,bins=int(np.floor(np.sqrt(len(calib_future_anomaly_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[2, 2].plot(future_anomaly_axis,kd_prior_future_anomaly(future_anomaly_axis),color=colors["prior"],label="prior")
    ax[2, 2].plot(future_anomaly_axis,target(future_anomaly_axis),color=colors["target"],label="target")
    ax[2, 2].plot(future_anomaly_axis,kd_calib_future_anomaly(future_anomaly_axis),color=colors["posterior_calib"])
    ax[2, 2].plot(future_anomaly_axis,posterior_mc(future_anomaly_axis),color=colors['posterior_mc'])
    ax[2, 2].set_xlim(0.7, 3.2)
    #ax[2, 2].set_ylim(0, 1.1)
    ax[2, 2].set_title("Future temperature anomaly")
    ax[2, 2].set_yticklabels([])
    ax[2, 2].set_xlabel("°C, 2081-2100 minus 1995-2014, ssp245")

    fig.tight_layout()
    if savefig:
        plt.savefig('figures/distributions/constraints')
    else:
        plt.show()