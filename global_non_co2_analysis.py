#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

import sys
cwd = os.path.abspath(os.path.join('..'))
fair_path = os.path.join(cwd, '/FAIR/src/')
if fair_path not in sys.path:
    sys.path.append(fair_path)
    
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
#from fair.earth_params import seconds_per_year

# datadir=Path('/Users/partanen/OneDrive - Ilmatieteen laitos/projects/POLKU/data')
# figdir=Path('/Users/partanen/OneDrive - Ilmatieteen laitos/projects/POLKU/figures')
# fairdir=Path('/Users/partanen/version-controlled/FAIR/src')
#fairdir=Path('/FAIR/')


def createRuns(scenarios=['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']):
    f = FAIR(ch4_method='thornhill2021')
    
    f.define_time(1750, 2100, 1)
    
    f.define_scenarios(scenarios)
    
    df = pd.read_csv("FAIR/tests/test_data/4xCO2_cummins_ebm3.csv")
    models = df['model'].unique()
    configs = []
    
    for imodel, model in enumerate(models):
        for run in df.loc[df['model']==model, 'run']:
            configs.append(f"{model}_{run}")
    f.define_configs(configs)
    
    
    species, properties = read_properties()
    #species = list(properties.keys())
    species[:5]
    properties['CO2 FFI']
    f.define_species(species, properties)
    
    f.allocate()
    
    f.fill_species_configs()
    fill(f.species_configs['unperturbed_lifetime'], 10.8537568, specie='CH4')
    fill(f.species_configs['baseline_emissions'], 19.01978312, specie='CH4')
    fill(f.species_configs['baseline_emissions'], 0.08602230754, specie='N2O')
    
    
    df_volcanic = pd.read_csv('FAIR/tests/test_data/volcanic_ERF_monthly_175001-201912.csv', index_col='year')
    f.fill_from_rcmip()
    
    # overwrite volcanic
    volcanic_forcing = np.zeros(351)
    volcanic_forcing[:271] = df_volcanic[1749:].groupby(int(np.ceil(df_volcanic[1749:].index))).mean().squeeze().values
    fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")  # sometimes need to expand the array
    
    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    
    df = pd.read_csv("FAIR/tests/test_data/4xCO2_cummins_ebm3.csv")
    models = df['model'].unique()
    
    seed = 1355763
    
    for config in configs:
        model, run = config.split('_')
        condition = (df['model']==model) & (df['run']==run)
        fill(f.climate_configs['ocean_heat_capacity'], df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
        fill(f.climate_configs['ocean_heat_transfer'], df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
        fill(f.climate_configs['deep_ocean_efficacy'], df.loc[condition, 'epsilon'].values[0], config=config)
        fill(f.climate_configs['gamma_autocorrelation'], df.loc[condition, 'gamma'].values[0], config=config)
        fill(f.climate_configs['sigma_eta'], df.loc[condition, 'sigma_eta'].values[0], config=config)
        fill(f.climate_configs['sigma_xi'], df.loc[condition, 'sigma_xi'].values[0], config=config)
        fill(f.climate_configs['stochastic_run'], False, config=config)
        fill(f.climate_configs['use_seed'], True, config=config)
        fill(f.climate_configs['seed'], seed, config=config)
    
        seed = seed + 399
    return f

# Default SSP1-1.9 run['ssp119']
scenarios = ['ssp126']
f_default = createRuns(scenarios=scenarios)
aerosol_species=['Sulfur', 'BC', 'OC',
       'NH3', 'NOx', 'VOC', 'CO', 'Aerosol-radiation interactions',
       'Aerosol-cloud interactions', 
       'Light absorbing particles on snow and ice']
co2_species=['CO2', 'CO2 AFOLU', 'CO2 FFI']
# Not sure what to do with these: 'Land use',


# SSP SSP1-1.9 run where non-CO2 GHG emissions are set to zero from 2020 onwards
f_non_co2_stop=createRuns(scenarios=scenarios)
for specie in f_non_co2_stop.emissions.specie:
    if specie not in co2_species+aerosol_species:
        f_non_co2_stop.emissions.loc[dict(specie=specie, timepoints=slice(2020,2200))]=0
        
# SSP1-1.9 run without aerosol forcing

f_no_aero=createRuns(scenarios=scenarios)
for specie in f_no_aero.emissions.specie:
    if specie in aerosol_species:
        f_no_aero.emissions.loc[dict(specie=specie)]=0


# SSP1-1.9 run non-CO2 GHGs
f_no_nonco2ghgs=createRuns(scenarios=scenarios)
for specie in f_no_nonco2ghgs.emissions.specie:
    if specie not in aerosol_species+co2_species:
        f_no_nonco2ghgs.emissions.loc[dict(specie=specie)]=0

# SSP1-1.9 run with only CO2 forcing
f_co2only=createRuns(scenarios=scenarios)
for specie in f_co2only.emissions.specie:
    if specie not in co2_species:
        f_co2only.emissions.loc[dict(specie=specie)]=0

f_default.run()
#f_non_co2_stop.run()
#f_no_aero.run()
#f_no_nonco2ghgs.run()
#f_co2only.run()

# Calculate aerosol contribution between 2020 and 2050
delta_T_aero=((f_default.temperature.mean(dim='config').sel(layer=0, timebounds=2050)-
              f_no_aero.temperature.mean(dim='config').sel(layer=0, timebounds=2050))-
              (f_default.temperature.mean(dim='config').sel(layer=0, timebounds=2020)-
               f_no_aero.temperature.mean(dim='config').sel(layer=0, timebounds=2020)))

# Calculate the change in non-CO2 GHG warming contribution from 1750-2020 emissions from 2020 to 2050 
delta_T_nonCO2_1750_2020=((f_non_co2_stop.temperature.mean(dim='config').sel(layer=0, timebounds=2050)-
              f_no_nonco2ghgs.temperature.mean(dim='config').sel(layer=0, timebounds=2050))-
              (f_non_co2_stop.temperature.mean(dim='config').sel(layer=0, timebounds=2020)-
                            f_no_nonco2ghgs.temperature.mean(dim='config').sel(layer=0, timebounds=2020)))

# Calculate the change in non-CO2 GHG warming contribution from 2020 to 2050 (from all emissions)
delta_T_nonCO2=((f_default.temperature.mean(dim='config').sel(layer=0, timebounds=2050)-
                 f_co2only.temperature.mean(dim='config').sel(layer=0, timebounds=2050))-
                (f_default.temperature.mean(dim='config').sel(layer=0, timebounds=2020)-
                            f_co2only.temperature.mean(dim='config').sel(layer=0, timebounds=2020)))
 

# Calculate allowable warming from GHG emissions after 2020
delta_T=0.43

# data.loc[dict(setup=setup_d,path=1)][score_var][:]=np.zeros(9)

specie = 'CO2'
specie_fig = plt.figure('specie')
ax_specie = specie_fig.gca()

T_fig = plt.figure('T')
ax_T = T_fig.gca()
for scenario in scenarios:
    ax_specie.plot(f_default.timebounds, f_default.concentration.loc[dict(scenario=scenario, specie=specie)], label=f_default.configs)
    ax_specie.set_title(f'{scenario}: {specie} concentration')
    ax_specie.set_xlabel('year')
    ax_specie.set_ylabel(f'{specie} concentration')
    
    ax_T.plot(f_default.timebounds, f_default.temperature.loc[dict(scenario=scenario, layer=0)], label=f_default.configs);
    ax_T.set_title(f'{scenario}: temperature trend')
    ax_T.set_xlabel('year')
    ax_T.set_ylabel('Temperature (°C)')
    
#ax_specie.legend()

