#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:06:51 2023

@author: nurmelaj
"""

import pandas as pd
import numpy as np
import xarray as xr
import warnings
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from pickle_tools import load_obj
import os
from dotenv import load_dotenv
cwd = os.getcwd()

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
fair_calibration_dir=f"{cwd}/fair-calibrate"

def load_data(scenario,nconfigs,start,end):
    '''
    Function to load input for a SSP2-4.5 run
    
    Parameters
    ----------
    scenario : str
        Chosen FaIR scenario.
    nconfigs : int
        Number of different configs for FaIR experiments
​
    Returns
    -------
    solar_forcing : Numpy array
        Solar forcing data
    volcanic_forcing : Numpy array
        Volcanic forcing data
    emissions : xarray DataArray
        Emissions for FaIR experiments
​
    '''
    df_solar = pd.read_csv(f"{fair_calibration_dir}/data/forcing/solar_erf_timebounds.csv",
                           index_col="year")
    df_volcanic = pd.read_csv(f"{fair_calibration_dir}/data/forcing/volcanic_ERF_monthly_-950001-201912.csv",
                              index_col='year')
    volcanic_forcing = np.zeros(end-start+1)
    df_volcanic = df_volcanic[(start-1):]
    #print(df_volcanic)
    #print(df_solar)
    #sys.exit()
    N = end - start + 1
    volcanic_forcing = np.zeros(N)
    max_index = int(np.ceil(df_volcanic.index[-1])) - start + 1
    volcanic_forcing[:max_index] = df_volcanic.groupby(np.ceil(df_volcanic.index.values)//1).mean().squeeze().values
    solar_forcing = df_solar["erf"].loc[start:end].values
    
    # run with harmonized emissions
    da_emissions = xr.load_dataarray(
        f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}"
        "/emissions/ssp_emissions_1750-2500.nc"
    )
    
    
    da = da_emissions.loc[dict(config="unspecified", scenario=scenario)][:N-1, ...]
    fe = da.expand_dims(dim=["scenario","config"], axis=(1,2))
    emissions = fe.drop("config") * np.ones((1, nconfigs, 1))
    
    return solar_forcing, volcanic_forcing, emissions

def load_configs():
    df = pd.read_csv(f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
                     "posteriors/calibrated_constrained_parameters.csv",
                     index_col=0)
    return df

def load_MC_configs(folder,scenario,names):
    filename = f'sampling_{scenario}'
    sampling = load_obj(f'{cwd}/{folder}',filename)
    mu, cov = sampling['mu'], sampling['cov']
    df = load_configs()
    samples = np.random.multivariate_normal(mu,cov,1001)
    df[names] = samples
    return df

def runFaIR(solar_forcing, volcanic_forcing, emissions, df_configs, scenario,
            start=1750, end=2020):
    '''
    Parameters
    ----------
    solar_forcing : Numpy array
        Solar forcing data
    volcanic_forcing : Numpy array
        Volcanic forcing data
    emissions : xarray DataArray
        Emissions for FaIR experiments
    df_configs : Pandas DataFrame
        Parameter dataframe. Each row correspond to one set of parameters.
    scenario : str
        FaIR scenario

    Returns
    -------
    f : FaIR run object
        
    Based on script fair-calibrate/input/fair-2.1.0/v1.0/GCP_2022/constraining/06_constrained-ssp-projections.py
    Modified e.g. to read in df_configs and shortened the run to 1750-2020
    '''
    
    valid_all = df_configs.index
    
    N = end - start + 1
    trend_shape = np.ones(N)
    trend_shape[:N] = np.linspace(0, 1, N)

    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(start, end, 1)
    f.define_scenarios([scenario])
    f.define_configs(valid_all)
    species, properties = read_properties()
    f.define_species(species, properties)
    f.allocate()

    # run with harmonized emissions
    f.emissions = emissions
    
    # solar and volcanic forcing
    fill(
        f.forcing,
        volcanic_forcing[:, None, None] * df_configs["scale Volcanic"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f.forcing,
        solar_forcing[:, None, None] * df_configs["solar_amplitude"].values.squeeze()
        + trend_shape[:, None, None] * df_configs["solar_trend"].values.squeeze(),
        specie="Solar",
    )
    
    # climate response
    fill(f.climate_configs["ocean_heat_capacity"], df_configs.loc[:, "c1":"c3"].values)
    fill(
        f.climate_configs["ocean_heat_transfer"],
        df_configs.loc[:, "kappa1":"kappa3"].values,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], df_configs["epsilon"].values.squeeze())
    fill(f.climate_configs["gamma_autocorrelation"], df_configs["gamma"].values.squeeze())
    fill(f.climate_configs["sigma_eta"], df_configs["sigma_eta"].values.squeeze())
    fill(f.climate_configs["sigma_xi"], df_configs["sigma_xi"].values.squeeze())
    fill(f.climate_configs["seed"], df_configs["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], df_configs["F_4xCO2"])
    
    # species level
    f.fill_species_configs()
    
    # carbon cycle
    fill(f.species_configs["iirf_0"], df_configs["r0"].values.squeeze(), specie="CO2")
    fill(
        f.species_configs["iirf_airborne"], df_configs["rA"].values.squeeze(), specie="CO2"
    )
    fill(f.species_configs["iirf_uptake"], df_configs["rU"].values.squeeze(), specie="CO2")
    fill(
        f.species_configs["iirf_temperature"],
        df_configs["rT"].values.squeeze(),
        specie="CO2",
    )
    
    # aerosol indirect
    fill(f.species_configs["aci_scale"], df_configs["beta"].values.squeeze())
    fill(
        f.species_configs["aci_shape"],
        df_configs["shape Sulfur"].values.squeeze(),
        specie="Sulfur",
    )
    fill(
        f.species_configs["aci_shape"], df_configs["shape BC"].values.squeeze(), specie="BC"
    )
    fill(
        f.species_configs["aci_shape"], df_configs["shape OC"].values.squeeze(), specie="OC"
    )
    
    # methane lifetime baseline - should be imported from calibration
    fill(f.species_configs["unperturbed_lifetime"], 10.11702748, specie="CH4")
    
    # emissions adjustments for N2O and CH4 (we don't want to make these defaults as people
    # might wanna run pulse expts with these gases)
    fill(f.species_configs["baseline_emissions"], 19.019783117809567, specie="CH4")
    fill(f.species_configs["baseline_emissions"], 0.08602230754, specie="N2O")
    
    # aerosol direct
    for specie in [
        "BC",
        "CH4",
        "N2O",
        "NH3",
        "NOx",
        "OC",
        "Sulfur",
        "VOC",
        "Equivalent effective stratospheric chlorine",
    ]:
        fill(
            f.species_configs["erfari_radiative_efficiency"],
            df_configs[f"ari {specie}"],
            specie=specie,
        )
    
    # forcing scaling
    for specie in [
        "CO2",
        "CH4",
        "N2O",
        "Stratospheric water vapour",
        "Contrails",
        "Light absorbing particles on snow and ice",
        "Land use",
    ]:
        fill(
            f.species_configs["forcing_scale"],
            df_configs[f"scale {specie}"].values.squeeze(),
            specie=specie,
        )
    
    for specie in [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1301",
        "Halon-2402",
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ]:
        fill(
            f.species_configs["forcing_scale"],
            df_configs["scale minorGHG"].values.squeeze(),
            specie=specie,
        )
    
    # ozone
    for specie in [
        "CH4",
        "N2O",
        "Equivalent effective stratospheric chlorine",
        "CO",
        "VOC",
        "NOx",
    ]:
        fill(
            f.species_configs["ozone_radiative_efficiency"],
            df_configs[f"o3 {specie}"],
            specie=specie,
        )
    
    # tune down volcanic efficacy
    fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")
    
    # land use parameter needs rescaling
    fill(
        f.species_configs["land_use_cumulative_emissions_to_forcing"],
        -0.000236847,
        specie="CO2 AFOLU",
    )
    
    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(
        f.species_configs["baseline_concentration"],
        df_configs["co2_concentration_1750"].values.squeeze(),
        specie="CO2",
    )
    
    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    f.run(progress=False)
    return f

def run_1pctco2(df_configs):
    scenarios = ["1pctCO2"]
    # batch_start = cfg["batch_start"]
    # batch_end = cfg["batch_end"]
    # batch_size = batch_end - batch_start

    species, properties = read_properties()

    da_concentration = xr.load_dataarray(
        f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "concentration/1pctCO2_concentration_1850-1990.nc"
    )

    f = FAIR()
    f.define_time(1850, 1990, 1)
    f.define_scenarios(scenarios)
    species = ["CO2", "CH4", "N2O"]
    properties = {
        "CO2": {
            "type": "co2",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "CH4": {
            "type": "ch4",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "N2O": {
            "type": "n2o",
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
    }
    valid_all = df_configs.index
    f.define_configs(valid_all)
    f.define_species(species, properties)
    f.allocate()

    da = da_concentration.loc[dict(config="unspecified", scenario="1pctCO2")]
    fe = da.expand_dims(dim=["scenario", "config"], axis=(1, 2))
    f.concentration = fe.drop("config") * np.ones((1, 1, len(valid_all), 1))

    # climate response
    fill(
        f.climate_configs["ocean_heat_capacity"],
        np.array([df_configs["c1"], df_configs["c2"], df_configs["c3"]]).T,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        np.array([df_configs["kappa1"], df_configs["kappa2"], df_configs["kappa3"]]).T,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], df_configs["epsilon"])
    fill(f.climate_configs["gamma_autocorrelation"], df_configs["gamma"])
    fill(f.climate_configs["stochastic_run"], False)
    fill(f.climate_configs["forcing_4co2"], df_configs["F_4xCO2"])

    # species level
    f.fill_species_configs()

    # carbon cycle
    
    # carbon cycle
    fill(f.species_configs["iirf_0"], df_configs["r0"].values.squeeze(), specie="CO2")
    fill(
        f.species_configs["iirf_airborne"], df_configs["rA"].values.squeeze(), specie="CO2"
    )
    fill(f.species_configs["iirf_uptake"], df_configs["rU"].values.squeeze(), specie="CO2")
    fill(
        f.species_configs["iirf_temperature"],
        df_configs["rT"].values.squeeze(),
        specie="CO2",
    )





    # forcing scaling
    fill(f.species_configs["forcing_scale"], df_configs["scale CO2"], specie="CO2")
    fill(f.species_configs["forcing_scale"], df_configs["scale CH4"], specie="CH4")
    fill(f.species_configs["forcing_scale"], df_configs["scale N2O"], specie="N2O")

    # initial conditions
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.run(progress=False)

    tcre=f.temperature.sel(layer=0, timebounds=1920)/f.cumulative_emissions.sel(specie='CO2', timebounds=1920)*1000*3.67
    tcr=f.temperature.sel(layer=0, timebounds=1920)

    # tcre_alt=

    return tcre,tcr