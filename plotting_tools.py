#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 2023

@author: nurmelaj
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
npr.seed(1234)
from copy import deepcopy
from pandas import concat,read_csv
from scipy.stats import gaussian_kde
from xarray import DataArray
# Import data reading tools
from fair_tools import compute_data_loss,compute_constraints,get_log_prior,setup_fair,run_configs
from fair_tools import get_constraint_ranges,get_log_constraint_target,get_constraint_targets,get_param_ranges,resample_constraint_posteriors
from fair_tools import temperature_anomaly
# dotenv parameters
from data_handling import fair_calib_dir,fair_v,constraint_set,cal_v,read_temperature_obs,read_forcing,read_emissions,transform_df

def plot_distributions(configs,configs_other=None,alpha=0.5,title=None,
                       figsize=(35,20),layout_shape=(9,5),savefig=True,colors=None):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #plt.ticklabel_format(useOffset=False)
    
    param_ranges = get_param_ranges()
    names = configs.columns
    names_other = configs_other.columns if configs_other is not None else []
    if layout_shape is None:
        layout_shape = get_layout_shape(max(len(names),len(names_other)))
    
    fig, axs = plt.subplots(layout_shape[0],layout_shape[1],figsize=figsize)
    for n in range(len(names)):
        name = names[n]
        row, col = (n - n % layout_shape[1])//layout_shape[1], n % layout_shape[1]
        axs[row][col].hist(configs[name].values,density=True,alpha=alpha,range=param_ranges[name],
                           bins=int(np.ceil(2*np.sqrt(len(configs)))),color=colors[0])
        if name in names_other:
            axs[row][col].hist(configs_other[name].values,density=True,alpha=alpha,range=param_ranges[name],
                               bins=int(np.ceil(2*np.sqrt(len(configs_other)))),color=colors[1])
        axs[row][col].set_title(name,fontsize=12)
    if configs_other is not None:
        extra_params = [param for param in names_other if param not in names]
        if len(extra_params) != 0:
            for n in range(len(names),len(names)+len(extra_params)):
                name = extra_params[n-len(names)]
                row, col = (n - n % layout_shape[1])//layout_shape[1], n % layout_shape[1]
                axs[row][col].hist(configs_other[name].values,density=True,alpha=alpha,range=param_ranges[name],
                                   bins=int(np.ceil(1.5*np.sqrt(len(configs_other)))),color=colors[1])
                axs[row][col].set_title(name,fontsize=12)
    for n in range(n+1, np.prod(layout_shape)):
        row, col = (n - n % layout_shape[1])//layout_shape[1], n % layout_shape[1]
        axs[row][col].axis('off')
    if title is not None:
        plt.suptitle(title,fontsize=16)
    fig.tight_layout()
    if savefig:
        save_name = title.lower().replace(' ', '_') if title is not None else 'distributions'
        fig.savefig(f"figures/distributions/{save_name}.png",dpi=300)
        fig.savefig(f"figures/distributions/{save_name}.pdf",dpi=300)
        plt.close()
    else:
        plt.show()
        
def plot_param_sensitivity(config,param_to_test,scenarios,param_range=None,include_prior=True,transformation_dict=None,
                           default_constraints=False,target_log_likelihood='wss',N=500,figsize=(10,8),savefig=False):
    if param_range is None:
        param_ranges = get_param_ranges()
        param_range = param_ranges[param_to_test]
    if target_log_likelihood == 'wss':
        obs, unc = read_temperature_obs()
        var = unc**2
    elif target_log_likelihood == 'trend':
        raise ValueError('Not implemented')
    if include_prior:
        log_prior = get_log_prior(config.columns[:-1])
    if default_constraints:
        log_constraint_target = get_log_constraint_target()

    solar_forcing, volcanic_forcing = read_forcing(1750,2023)
    configs = concat([config]*N, ignore_index=True, axis='rows')
    for scenario in scenarios:
        configs[param_to_test] = np.linspace(min(param_range),max(param_range),N)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        fair_allocated = setup_fair([scenario],N,start=1750,end=2023)
        fair = run_configs(deepcopy(fair_allocated),
                           transform_df(configs,transformation_dict, 
                                        name_changes = {'log(-aci_beta)': 'aci_beta',
                                                        'log(aci_shape_SO2)': 'aci_shape_SO2',
                                                        'log(aci_shape_BC)': 'aci_shape_BC',
                                                        'log(aci_shape_OC)': 'aci_shape_OC'}),
                           solar_forcing,volcanic_forcing,start=1750,end=2023,stochastic_run=False)
        model_T = temperature_anomaly(fair,start=1850,end=2023,rolling_window_size=2)
        loss = compute_data_loss(model_T,obs,var,target=target_log_likelihood)
        if include_prior:
            loss += DataArray(data=-log_prior(configs[config.columns[:-1]].to_numpy()),
                              dims=['config'],coords=dict(config=('config',configs.index)))
        if default_constraints:
            loss += DataArray(data=-log_constraint_target(compute_constraints(fair)),
                              dims=['scenario','config'],
                              coords=dict(scenario=('scenario',[scenario]),config=('config',configs.index)))
        ax.plot(configs[param_to_test].values,loss.sel(scenario=scenario).data,'k-')
        ax.set_title(param_to_test,fontsize=14)
        if savefig:
            fig.savefig(f'figures/sensitivity/{scenario}_{param_to_test}_{target_log_likelihood}_sensitivity.png',dpi=300)
            fig.savefig(f'figures/sensitivity/{scenario}_{param_to_test}_{target_log_likelihood}_sensitivity.pdf',dpi=300)
        else:
            plt.show()
        
def sensitivity_test(config,included,scenarios,param_ranges=None,ylims=None,default_constraints=True,include_prior=True,
                     transformation_dict=None,target_log_likelihood='wss',layout_shape=None,N=500,savefig=True,figsize=(35,25)):
    if param_ranges is None:
        param_ranges = get_param_ranges()
    '''
    if transformation_dict is None and transformation_logabsjacdet is not None:
        print('Logarithm of the Jacobian determinant is ignored when no transformation is given')
        transformation_logabsjacdet = lambda x: 0.0
    elif transformation_dict is not None and transformation_logabsjacdet is None:
        raise ValueError('Analytic expression for the logarithm of the Jacobian determinant of the transformation must be provided')
    '''
    if target_log_likelihood == 'wss':
        obs, unc = read_temperature_obs()
        var = unc**2
    elif target_log_likelihood == 'trend':
        raise ValueError('Not implemented')
        #trends = compute_trends()
        #obs, var = np.mean(trends), np.std(trends)**2

    solar_forcing, volcanic_forcing = read_forcing(1750,2023)
    configs = concat([config]*N, ignore_index=True, axis='rows')
    #configs['seed'] = npr.randint(min(param_ranges['seed']),max(param_ranges['seed']),size=N)
    #configs['seed'] = 1234
    if include_prior:
        log_prior = get_log_prior(included)
    if default_constraints:
        log_constraint_target = get_log_constraint_target()
    if ylims is None:
        ylims_dict = {}
    if layout_shape is None:
        layout_shape = get_layout_shape(len(included))
    fig, axs = plt.subplots(layout_shape[0],layout_shape[1],figsize=figsize)
    #plt.ticklabel_format(useOffset=False)
    for scenario in scenarios:
        emissions = read_emissions(1750,2023,[scenario],nconfigs=N)
        fair_allocated = setup_fair([scenario],N,emissions,start=1750,end=2023)
        for n in range(len(included)):
            param = included[n]
            print(param)
            # List of different values for a parameter
            if param == 'clim_c2':
                configs[param] = np.linspace(config['clim_c1'].iloc[0],max(param_ranges[param]),N)
            elif param == 'clim_c3':
                configs[param] = np.linspace(config['clim_c2'].iloc[0],max(param_ranges[param]),N)
            else:
                configs[param] = np.linspace(min(param_ranges[param]),max(param_ranges[param]),N)
            fair = run_configs(deepcopy(fair_allocated),
                               transform_df(configs,transformation_dict, 
                                            name_changes = {'log(-aci_beta)': 'aci_beta',
                                                            'log(aci_shape_SO2)': 'aci_shape_SO2',
                                                            'log(aci_shape_BC)': 'aci_shape_BC',
                                                            'log(aci_shape_OC)': 'aci_shape_OC'}),
                               solar_forcing,volcanic_forcing,start=1750,end=2023,stochastic_run=False)
            model_T = temperature_anomaly(fair,start=1850,end=2023,rolling_window_size=2)
            loss = compute_data_loss(model_T,obs,var,target=target_log_likelihood)
            if include_prior:
                loss += DataArray(data=-log_prior(configs[included].to_numpy()),
                                  dims=['config'],coords=dict(config=('config',configs.index)))
            if default_constraints:
                loss += DataArray(data=-log_constraint_target(compute_constraints(fair)),
                                  dims=['scenario','config'],
                                  coords=dict(scenario=('scenario',[scenario]),config=('config',configs.index)))
            row, col = (n - n % layout_shape[1])//layout_shape[1], n % layout_shape[1]
            axs[row][col].plot(configs[param].values,loss.sel(scenario=scenario).data,'k-')
            if ylims is not None:
                axs[row][col].set_ylim(min(ylims),max(ylims))
            else:
                ylims_dict[param] = [loss.sel(scenario=scenario).min(dim='config'),
                                     loss.sel(scenario=scenario).max(dim='config')]
            #axs[row][col].set_xlabel(param)
            axs[row][col].set_title(param,fontsize=14)
            axs[row][col].set_yticks([],[])
            # Reset param
            configs[param] = np.repeat(float(config[param].iloc[0]),N)
        for n in range(n+1,np.prod(layout_shape)):
            row, col =  (n - n % layout_shape[1])//layout_shape[1], n % layout_shape[1]
            axs[row][col].axis('off')

        if ylims is None:
            minlims = np.array([min(ylims_dict[param]) for param in included])
            maxlims = np.array([max(ylims_dict[param]) for param in included])
            minlims = minlims[~np.isinf(minlims) & ~np.isnan(minlims)]
            maxlims = maxlims[~np.isinf(maxlims) & ~np.isnan(maxlims)]
            # Minimum of all lower limits
            min_min = np.min(minlims)
            # Maximum of all lower limits
            min_max = np.max(minlims)
            # Median of all lower limits
            # min_median = np.median(minlims)
            # Common bounds for every plot
            ymin, ymax = min_min, min_max + np.median(maxlims - minlims)
            
            #ymin = min_median - 1.1 * (min_median-min_min)
            #ymax = min_median + 1.5 * (min_median-min_min)
            for n in range(len(included)):
                row, col = (n - n % layout_shape[1])//layout_shape[1], n % layout_shape[1]
                axs[row][col].set_ylim(ymin,ymax)
        fig.tight_layout()
        #plt.suptitle(f'Sentivity test, {data_loss_method} loss function',fontsize=24)
        if savefig:
            fig.savefig(f'figures/sensitivity/{scenario}_loss_sensitivity_{target_log_likelihood}.png',dpi=300)
            fig.savefig(f'figures/sensitivity/{scenario}_loss_sensitivity_{target_log_likelihood}.pdf',dpi=300)
        else:
            plt.show()
    
def get_layout_shape(N,threshold=3):
    '''
    Find suitable layout for N number of images.
    '''
    if np.sqrt(N).is_integer():
        return (int(np.sqrt(N)),int(np.sqrt(N)))
    elif N in [2,3,5]:
        return (N,1)
    else:
        while True:
            s = int(np.ceil(np.sqrt(N)))
            rows = s
            while rows - s <= threshold:
                for cols in range(rows-threshold,rows):
                    if rows*cols == N:
                        return (rows,cols)
                rows += 1
            N += 1
    
def plot_chains(chain,params=None,figsize=(30,25)):
    if params is None:
        params = [f'x_{n+1}' for n in range(chain.shape[1])]
    if len(params) != chain.shape[1]:
        raise ValueError(f'Length of parameter names {len(params)} differs from chain dimension {chain.shape[1]}')
    N = len(chain)
    shape = get_layout_shape(len(params))
    fig, axs = plt.subplots(shape[1],shape[0],figsize=figsize)
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
    
def plot_temperature(fair_run,fair_other_run=None,start=1750,end=2100,MAP=None,obs=None,obs_std=None,
                     plot_trend=False,savefig=True,labels=None,colors=None):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    years = np.arange(start,end+1,dtype=np.uint32)
    if labels is not None and fair_other_run is not None and obs is not None:
        first_label, second_label, third_label = labels
    elif labels is not None and fair_other_run is not None and obs is None:
        first_label, second_label = labels
    elif labels is not None and fair_other_run is None and obs is not None:
        first_label, third_label = labels
    elif fair_other_run is None and obs is None and labels is not None:
        first_label = labels[0] if isinstance(labels,list) or isinstance(labels,tuple) else labels
    elif fair_other_run is not None and obs is not None and labels is None:
        first_label, second_label, third_label = None, None, None
    elif fair_other_run is not None and obs is None and labels is None:
        first_label, second_label = None, None
    else:
        first_label, third_label = None, None
    for scenario in fair_run.scenarios:
        N_plot = 0
        T_fig = plt.figure('T_figure',figsize=(8,4))
        ax_T = T_fig.gca()
        #ax_T.set_title(f'Temperature trend, scenario {scenario}')
        ax_T.set_xlabel('year')
        ax_T.set_ylabel('Temperature (°C)')
        ax_T.set_ylim(-1.5,4)
        
        if start > 1850 or end < 1901:
            raise ValueError('Start year can not be later than 1850 and end date con not be before 1901')
        # Reference temperature as the average between the temperatures between 1850-1901, weights equal to 1.0 for each year
        # except the start and end years which have weight 0.5
        ref_years = np.arange(1850,1901,dtype=np.uint32)
        weights_50yr = DataArray(data=np.concatenate(([0.5],np.ones(49),[0.5]))/50,dims=['timebounds'],coords=dict(timebounds=("timebounds",ref_years)))              
        ref_T = fair_run.temperature.sel(timebounds=ref_years,scenario=scenario,layer=0).weighted(weights=weights_50yr).sum(dim='timebounds')
        model_T = fair_run.temperature.sel(timebounds=years,scenario=scenario,layer=0) - ref_T
        mean = model_T.mean(dim='config')
        std = model_T.std(dim='config')
        ax_T.plot(years, mean, linestyle='-', color=colors[N_plot], label=first_label)
        ax_T.fill_between(years,mean-1.96*std,mean+1.96*std,color=colors[N_plot],alpha=0.4,label='95% confidence' if first_label is not None else None)
        N_plot += 1
        
        if fair_other_run is not None:
            ref_T = fair_other_run.temperature.sel(timebounds=ref_years,scenario=scenario,layer=0).weighted(weights=weights_50yr).sum(dim='timebounds')
            model_T = fair_other_run.temperature.sel(timebounds=slice(start,end),scenario=scenario,layer=0) - ref_T
            mean = model_T.mean(dim='config')
            std = model_T.std(dim='config')
            ax_T.plot(years, mean, linestyle='-', color=colors[N_plot], markersize=2, label=second_label)
            ax_T.fill_between(years,mean-1.96*std,mean+1.96*std,color=colors[N_plot],alpha=0.4)
            N_plot += 1
            
        if obs is not None:
            years = 0.5*np.round(2*obs.time.data)
            #years = np.array(map(lambda timestamp: datetime.fromtimestamp(timestamp/int(1e9)), obs.time.data.astype(int)),dtype=np.int32)
            ax_T.plot(years, obs, linestyle='-', color=colors[N_plot], label=third_label)
            if obs_std is not None:
                ax_T.fill_between(years,obs.data-1.96*obs_std.data,obs.data+1.96*obs_std.data,
                                  color=colors[N_plot],alpha=0.4)
            if plot_trend:
                x = np.arange(1900,2021,1)
                y = obs.sel(year=slice(1900,2020)).data
                params = np.polyfit(x,y,1)
                ax_T.plot(x,params[0] * x + params[1], label='trend line 1900-2020', color='black')
            N_plot += 1
            
        if MAP is not None:
            MAP_temperature = MAP.temperature.sel(timebounds=slice(start,end),scenario=scenario,layer=0) \
                            - MAP.temperature.sel(timebounds=slice(1851,1901),scenario=scenario,layer=0).mean(dim='timebounds').to_numpy()
            ax_T.plot(years, MAP_temperature.data.squeeze(), linestyle='-', color='black', markersize=2, label='MAP')
        if first_label is not None:
            ax_T.legend(loc='upper left')
        if savefig:
            plt.savefig(f'figures/trends/temperature_{scenario}_{start}-{end}.png',dpi=300)
            plt.savefig(f'figures/trends/temperature_{scenario}_{start}-{end}.pdf',dpi=300)
            #plt.savefig(f'figures/stochastic_vs_deterministic_{min(years)}-{max(years)}')
        else:
            plt.show()
        plt.close()
    
def plot_specie(fair_run,fair_other_run=None,start=1750,end=2100,savefig=True,labels=None):
    '''
    Plots all species for different scenario in the fair run.
    '''
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if fair_other_run is not None and labels is not None:
        first_label, second_label = labels
    elif fair_other_run is None and labels is not None:
        first_label = labels[0] if isinstance(labels,list) or isinstance(labels,tuple) else labels
    elif fair_other_run is not None and labels is None:
        first_label, second_label = None, None
    else:
        first_label = None
        
    for specie in fair_run.species:
        for scenario in fair_run.scenarios:
            specie_fig = plt.figure(figsize=(10,5))
            ax_specie = specie_fig.gca()
            years = np.arange(start,end+1,dtype=np.int32)
            # Realization for different configurations
            realizations = fair_run.concentration.sel(timebounds=years,scenario=scenario,specie=specie)
            mean = realizations.mean(dim='config')
            std = realizations.std(dim='config')
            ax_specie.plot(years, mean, linestyle='-',color=colors[0],label=first_label)
            ax_specie.fill_between(years,mean-1.96*std,mean+1.96*std,color=colors[0],alpha=0.4)
            if fair_other_run is not None:
                realizations = fair_other_run.concentration.sel(timebounds=years,secnario=scenario,specie=specie)
                mean = realizations.mean(dim='config')
                std = realizations.std(dim='config')
                ax_specie.plot(years, mean, linestyle='-', color=colors[1], label=second_label)
                ax_specie.fill_between(std.year,mean-1.96*std,mean+1.96*std,color=colors[1],alpha=0.4)
            ax_specie.set_title(f'{scenario}, specie {specie} concentration')
            ax_specie.set_xlabel('year')
            ax_specie.set_ylabel('concentration')
            if savefig:
                plt.savefig(f'figures/trends/{scenario}_{specie}_concentration')
                plt.close(specie_fig)
            else:
                plt.show()
'''
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
'''

def plot_constraints(sampling=None,resample=False,plot_histo=True,alpha=0.6,colors=None,savefig=True):
    if colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {'prior': '#207F6E', 'posterior_calib': default_colors[0], 
                  'posterior_mc': default_colors[1], 'target': 'black'}
    
    # Constraint ranges (min,max) and target densities
    constraint_ranges = get_constraint_ranges()
    constraint_targets = get_constraint_targets()
    # Prior data
    prior_ecs_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy")
    prior_tcr_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy")
    prior_ohc_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ocean_heat_content_2020_minus_1971.npy") / 1e21
    temp = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy")
    weights_20yr = np.concatenate(([0.5],np.ones(19),[0.5])) / 20
    weights_50yr = np.concatenate(([0.5],np.ones(49),[0.5])) / 50
    prior_T_2003_2022_samples = np.average(temp[153:174,:], weights=weights_20yr, axis=0) \
                              - np.average(temp[:51,:], weights=weights_50yr, axis=0)
    prior_ari_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ari_2005-2014_mean.npy")
    prior_aci_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_aci_2005-2014_mean.npy")
    prior_aer_samples = prior_aci_samples + prior_ari_samples
    prior_co2_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/concentration_co2_2022.npy")
    prior_Tvar_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/T_variation_1750-2022.npy")
    prior_Tvar_samples = prior_Tvar_samples[np.isfinite(prior_Tvar_samples)]
    
    if resample:
        print('Resampling...')
        constraint_samples = resample_constraint_posteriors()
        print('...done')
    else:
        constraint_samples = read_csv(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/resampled_constraint_posteriors.csv")
    ecs_samples = constraint_samples['ECS'].to_numpy()
    tcr_samples = constraint_samples['TCR'].to_numpy()
    ohc_samples = constraint_samples['OHC'].to_numpy()
    T_2003_2022_samples = constraint_samples['T 2003-2022'].to_numpy()
    erfari_samples = constraint_samples['ERFari'].to_numpy()
    erfaci_samples = constraint_samples['ERFaci'].to_numpy()
    erfaer_samples = constraint_samples['ERFaer'].to_numpy()
    co2_samples = constraint_samples['CO2 conc 2022'].to_numpy()
    #Tvariation_samples = constraint_samples['Tvar'].to_numpy()
    
    if sampling is not None:
        #samples = sampling.dimensions['sample'].size - int(sampling['warmup'][:].data)
        #required_samples = N * thinning
        samples, parallel_chains, variables = sampling.dimensions['sample'].size, sampling.dimensions['chain_id'].size, sampling.dimensions['constraint'].size
        constraint_samples = sampling['constraints'][:,:,:].data.reshape((samples*parallel_chains,variables))
        constraints = list(sampling['constraint'][:])
        mc_ecs_samples = constraint_samples[:,constraints.index('ECS')]
        mc_tcr_samples = constraint_samples[:,constraints.index('TCR')]
        mc_Tincrease_samples = constraint_samples[:,constraints.index('T 2003-2022')]
        mc_ari_samples = constraint_samples[:,constraints.index('ERFari')]
        mc_aci_samples = constraint_samples[:,constraints.index('ERFaci')]
        mc_aer_samples = constraint_samples[:,constraints.index('ERFaer')]
        mc_co2_samples = constraint_samples[:,constraints.index('CO2 conc 2022')]
        mc_ohc_samples = constraint_samples[:,constraints.index('OHC')]
        mc_Tvar_samples = constraint_samples[:,constraints.index('Tvar')]
        mc_Tvar_samples = mc_Tvar_samples[np.isfinite(mc_Tvar_samples)]
    
    # Initiate figure
    print('Plotting constraints...')
    fig, ax = plt.subplots(3, 3, figsize=(15,10))
    
    print("Equilibrium Climate Sensitivity")
    ecs_axis = np.linspace(min(constraint_ranges['ECS']),max(constraint_ranges['ECS']),500)
    target = constraint_targets['ECS']
    
    kd_prior_ecs = gaussian_kde(prior_ecs_samples)
    kd_calib_ecs = gaussian_kde(ecs_samples)
    
    if plot_histo:
        #ax[0,0].hist(prior_ecs_samples,bins=int(np.floor(np.sqrt(len(prior_ecs_samples)))),density=True,color=colors['prior'])
        ax[0,0].hist(ecs_samples,bins=int(np.floor(np.sqrt(len(ecs_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[0,0].plot(ecs_axis,kd_prior_ecs(ecs_axis),color=colors["prior"],label="prior")
    ax[0,0].plot(ecs_axis,target(ecs_axis),color=colors["target"],label="target")
    ax[0,0].plot(ecs_axis,kd_calib_ecs(ecs_axis),color=colors['posterior_calib'])
    if sampling is not None:
        if plot_histo:
            ax[0,0].hist(mc_ecs_samples,bins=int(np.floor(np.sqrt(len(mc_ecs_samples)))),density=True,
                          color=colors['posterior_mc'],alpha=alpha,label='pos MC')
        kd_mc = gaussian_kde(mc_ecs_samples)
        ax[0,0].plot(ecs_axis,kd_mc(ecs_axis),color=colors['posterior_mc'])
    ax[0,0].set_xlim(1, 7)
    ax[0,0].set_title("ECS")
    ax[0,0].set_yticklabels([])
    ax[0,0].set_xlabel("°C")
    
    print("Transient Climate Response")
    tcr_axis = np.linspace(min(constraint_ranges['TCR']),max(constraint_ranges['TCR']),500)
    target = constraint_targets['TCR']
    kd_prior_tcr = gaussian_kde(prior_tcr_samples)
    kd_tcr = gaussian_kde(tcr_samples)
    if plot_histo:
        #ax[0,1].hist(prior_tcr_samples,bins=int(np.floor(np.sqrt(len(prior_tcr_samples)))),density=True,color=colors['prior'])
        ax[0,1].hist(tcr_samples,bins=int(np.floor(np.sqrt(len(tcr_samples)))),density=True,
                     color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[0,1].plot(tcr_axis,kd_prior_tcr(tcr_axis),color=colors["prior"],label="prior")
    ax[0,1].plot(tcr_axis,target(tcr_axis),color=colors["target"],label="target")
    ax[0,1].plot(tcr_axis,kd_tcr(tcr_axis),color=colors['posterior_calib'])
    if sampling is not None:
        if plot_histo:
            ax[0,1].hist(mc_tcr_samples,bins=int(np.floor(np.sqrt(len(mc_tcr_samples)))),density=True,
                          color=colors['posterior_mc'],alpha=alpha,label='pos MC')
        kd_mc = gaussian_kde(mc_tcr_samples)
        ax[0,1].plot(tcr_axis,kd_mc(tcr_axis),color=colors['posterior_mc'])
    ax[0,1].set_xlim(0.5, 3.5)
    ax[0,1].set_title("Transient Climate Response")
    ax[0,1].set_yticklabels([])
    ax[0,1].set_xlabel("°C")
    
    print('Temperature increase from 1850-1900 mean to 2003-2022 mean')
    T_axis = np.linspace(min(constraint_ranges['T 2003-2022']),max(constraint_ranges['T 2003-2022']),500)
    kd_prior_T = gaussian_kde(prior_T_2003_2022_samples)
    kd_T = gaussian_kde(T_2003_2022_samples)
    target = constraint_targets['Tinc']
    if plot_histo:
        ax[0,2].hist(prior_T_2003_2022_samples,bins=int(np.floor(np.sqrt(len(prior_T_2003_2022_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[0,2].plot(T_axis,kd_prior_T(T_axis),color=colors["prior"],label="prior")
    ax[0,2].plot(T_axis,target(T_axis),color=colors["target"],label="target")
    ax[0,2].plot(T_axis,kd_T(T_axis),color=colors["posterior_calib"])
    if sampling is not None:
        if plot_histo:
            ax[0,2].hist(mc_Tincrease_samples,bins=int(np.floor(np.sqrt(len(mc_Tincrease_samples)))),
                         density=True,color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_Tincrease_samples)
        ax[0,2].plot(T_axis,kd_mc(T_axis),color=colors['posterior_mc'])
    ax[0,2].set_xlim(0.14,2.33)
    #ax[2, 1].set_ylim(0, 0.006)
    ax[0,2].set_title("T 2003-2022 minus 1850-1900")
    ax[0,2].set_yticklabels([])
    ax[0,2].set_xlabel("°C")
    
    print('ERFari')
    ari_axis = np.linspace(min(constraint_ranges['ERFari']),max(constraint_ranges['ERFari']),500)
    kd_prior_ari = gaussian_kde(prior_ari_samples)
    kd_erfari = gaussian_kde(erfari_samples)
    target = constraint_targets['ERFari']
    if plot_histo:
        # ax[1,0].hist(prior_ari_samples,bins=int(np.floor(np.sqrt(len(prior_ari_samples)))),density=True,color=colors['prior'])
        ax[1,0].hist(erfari_samples,bins=int(np.floor(np.sqrt(len(erfari_samples)))),density=True,
                     color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[1,0].plot(ari_axis,kd_prior_ari(ari_axis),color=colors["prior"],label="prior")
    ax[1,0].plot(ari_axis,target(ari_axis),color=colors["target"],label="target")
    ax[1,0].plot(ari_axis,kd_erfari(ari_axis),color=colors['posterior_calib'])
    if sampling is not None:
        if plot_histo:
            ax[1,0].hist(mc_ari_samples,bins=int(np.floor(np.sqrt(len(mc_ari_samples)))),density=True,
                         color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_ari_samples)
        ax[1,0].plot(ari_axis,kd_mc(ari_axis),color=colors['posterior_mc'])
    ax[1,0].set_xlim(-1.0, 0.4)
    ax[1,0].set_ylim(0.0, 3.0)
    ax[1,0].set_title("ERFari 2005-2014 mean")
    ax[1,0].set_yticklabels([])
    ax[1,0].set_xlabel("W/m$^{2}$")
    
    print('ERFaci')
    aci_axis = np.linspace(min(constraint_ranges['ERFaci']),max(constraint_ranges['ERFaci']),500)
    kd_prior_aci = gaussian_kde(prior_aci_samples)
    kd_erfaci = gaussian_kde(erfaci_samples)
    target = constraint_targets['ERFaci']
    if plot_histo:
        # ax[1,0].hist(prior_ari_samples,bins=int(np.floor(np.sqrt(len(prior_ari_samples)))),density=True,color=colors['prior'])
        ax[1,1].hist(erfaci_samples,bins=int(np.floor(np.sqrt(len(erfaci_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[1,1].plot(aci_axis,kd_prior_aci(aci_axis),color=colors["prior"],label="prior")
    ax[1,1].plot(aci_axis,target(aci_axis),color=colors["target"],label="target")
    if sampling is not None:
        if plot_histo:
            ax[1,1].hist(mc_aci_samples,bins=int(np.floor(np.sqrt(len(mc_aci_samples)))),density=True,
                         color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_aci_samples)
        ax[1,1].plot(aci_axis,kd_mc(aci_axis),color=colors['posterior_mc'])
    ax[1,1].plot(aci_axis,kd_erfaci(aci_axis),color=colors['posterior_calib'])
    ax[1,1].set_xlim(-2.4, 0.3)
    ax[1,1].set_ylim(0.0, 3.0)
    ax[1,1].set_title("ERFaci 2005-2014 mean")
    ax[1,1].set_yticklabels([])
    ax[1,1].set_xlabel("W/m$^{2}$")
    
    print('ERFaer')
    aer_axis = np.linspace(min(constraint_ranges['ERFaer']),max(constraint_ranges['ERFaer']),500)
    kd_prior_aer = gaussian_kde(prior_aer_samples)
    kd_erfaer = gaussian_kde(erfaer_samples)
    target = constraint_targets['ERFaer']
    if plot_histo:
        # ax[1,0].hist(prior_ari_samples,bins=int(np.floor(np.sqrt(len(prior_ari_samples)))),density=True,color=colors['prior'])
        ax[1,2].hist(erfaer_samples,bins=int(np.floor(np.sqrt(len(erfaer_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[1,2].plot(aer_axis,kd_prior_aer(aer_axis),color=colors["prior"],label="prior")
    ax[1,2].plot(aer_axis,target(aer_axis),color=colors["target"],label="target")
    ax[1,2].plot(aer_axis,kd_erfaer(aer_axis),color=colors['posterior_calib'])
    if sampling is not None:
        if plot_histo:
            ax[1,2].hist(mc_aer_samples,bins=int(np.floor(np.sqrt(len(mc_aer_samples)))),density=True,
                         color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_aer_samples)
        ax[1,2].plot(aer_axis,kd_mc(aer_axis),color=colors['posterior_mc'])
    ax[1,2].set_xlim(-3.0, 0.2)
    ax[1,2].set_ylim(0.0, 3.0)
    ax[1,2].set_title("ERFaer 2005-2014 mean")
    ax[1,2].set_yticklabels([])
    ax[1,2].set_xlabel("W/m$^{2}$")
    
    print('CO2 concentration 2022')
    co2_axis = np.linspace(min(constraint_ranges['CO2 conc 2022']),max(constraint_ranges['CO2 conc 2022']),500)
    kd_prior_co2 = gaussian_kde(prior_co2_samples)
    kd_co2 = gaussian_kde(co2_samples)
    target = constraint_targets['CO2 conc 2022']
    if plot_histo:
        # ax[2, 0].hist(prior_co2_samples,bins=int(np.floor(np.sqrt(len(prior_co2_samples)))),density=True,color=colors['prior'])
        ax[2,0].hist(co2_samples,bins=int(np.floor(np.sqrt(len(co2_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[2,0].plot(co2_axis,kd_prior_co2(co2_axis),color=colors["prior"],label="prior")
    ax[2,0].plot(co2_axis,target(co2_axis),color=colors["target"],label="target")
    ax[2,0].plot(co2_axis,kd_co2(co2_axis),color=colors["posterior_calib"])
    if sampling is not None:
        if plot_histo:
            ax[2,0].hist(mc_co2_samples,bins=int(np.floor(np.sqrt(len(mc_co2_samples)))),density=True,
                          color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_co2_samples)
        ax[2,0].plot(co2_axis,kd_mc(co2_axis),color=colors['posterior_mc'])
    ax[2,0].set_xlim(414.0, 450.0)
    ax[2,0].set_title("CO$_2$ conc 2022")
    ax[2,0].set_yticklabels([])
    ax[2,0].set_xlabel("ppm")
    
    
    print('OHC change')
    ohc_axis = np.linspace(min(constraint_ranges['OHC']),max(constraint_ranges['OHC']),500)
    kd_prior_ohc = gaussian_kde(prior_ohc_samples)
    kd_ohc = gaussian_kde(ohc_samples)
    target = constraint_targets['OHC']
    
    if plot_histo:
        # ax[2, 1].hist(prior_ohc_samples,bins=int(np.floor(np.sqrt(len(prior_ohc_samples)))),density=True,color=colors['prior'])
        ax[2,1].hist(ohc_samples,bins=int(np.floor(np.sqrt(len(ohc_samples)))),density=True,
                      color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[2,1].plot(ohc_axis,kd_prior_ohc(ohc_axis),color=colors["prior"],label="prior")
    ax[2,1].plot(ohc_axis,target(ohc_axis),color=colors["target"],label="target")
    ax[2,1].plot(ohc_axis,kd_ohc(ohc_axis),color=colors["posterior_calib"])
    if sampling is not None:
        if plot_histo:
            ax[2,1].hist(mc_ohc_samples,bins=int(np.floor(np.sqrt(len(mc_ohc_samples)))),density=True,
                          color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_ohc_samples)
        ax[2,1].plot(ohc_axis,kd_mc(ohc_axis),color=colors['posterior_mc'])
    ax[2,1].set_xlim(0, 900)
    ax[2,1].set_title("OHC 2020 minus 1971")
    ax[2,1].set_yticklabels([])
    ax[2,1].set_xlabel("$10^{21}$ J")
    
    print('Temperature variation')
    Tvar_axis = np.linspace(min(constraint_ranges['Tvar']),max(constraint_ranges['Tvar']),500)
    kd_prior_Tvar = gaussian_kde(prior_Tvar_samples)
    #kd_Tvar = gaussian_kde(Tvariation_samples)
    target = constraint_targets['Tvar']
    if plot_histo:
        pass
        #ax[2,2].hist(Tvariation_samples,bins=int(np.floor(np.sqrt(len(Tvariation_samples)))),density=True,
        #              color=colors['posterior_calib'],label='pos calib',alpha=alpha)
    ax[2,2].plot(Tvar_axis,kd_prior_Tvar(Tvar_axis),color=colors["prior"],label="prior")
    ax[2,2].plot(Tvar_axis,target(Tvar_axis),color=colors["target"],label="target")
    #ax[2,2].plot(Tvar_axis,kd_Tvar(Tvar_axis),color=colors["posterior_calib"])
    if sampling is not None:
        if plot_histo:
            ax[2,2].hist(mc_Tvar_samples,bins=int(np.floor(np.sqrt(len(mc_co2_samples)))),density=True,
                         color=colors['posterior_mc'],label='pos MC',alpha=alpha)
        kd_mc = gaussian_kde(mc_Tvar_samples)
        ax[2,2].plot(Tvar_axis,kd_mc(Tvar_axis),color=colors['posterior_mc'])
    ax[2,2].set_xlim(0.06, 0.18)
    #ax[2,1].set_ylim(0, 1.2)
    ax[2,2].set_title("T variation 1850-2024")
    ax[2,2].set_yticklabels([])
    ax[2,2].set_xlabel("°C")
    
    # Set legend to the center figure
    ax[1,1].legend(frameon=False)
    fig.tight_layout()
    
    if savefig:
        fig.savefig('figures/distributions/constraints.pdf',dpi=300)
    else:
        plt.show()