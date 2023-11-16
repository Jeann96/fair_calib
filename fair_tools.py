#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 2023

@author: nurmelaj
"""

import pandas as pd
import numpy as np
import numpy.random as npr
import xarray as xr
import warnings
from scipy.stats import norm, uniform, skewnorm, gaussian_kde
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
import os
cwd = os.getcwd()
#from dotenv import load_dotenv
#load_dotenv()
#cal_v = os.getenv("CALIBRATION_VERSION")
#fair_v = os.getenv("FAIR_VERSION")
#constraint_set = os.getenv("CONSTRAINT_SET")
#output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
#plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")

fair_calibration_dir=f"{cwd}/fair-calibrate"

def load_dotenv():
    f = open('.env','r')
    out = {}
    for line in f:
        if '=' not in line:
            continue
        variable, value = line.split(' ')[0].split('=')
        out[variable] = value.strip('\n')
    f.close()
    return out

out = load_dotenv()
cal_v = out["CALIBRATION_VERSION"]
fair_v = out["FAIR_VERSION"]
constraint_set = out["CONSTRAINT_SET"]
output_ensemble_size = int(out["POSTERIOR_SAMPLES"])

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
    solar_forcing : nd-array
        Solar forcing data
    volcanic_forcing : nd-array
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
    
    #df_volcanic = df_volcanic[(start-1):]
    df_volcanic = df_volcanic[(start - 1 ) < df_volcanic.index]
    L = end - start + 1
    volcanic_forcing = np.zeros(L)
    # Last year in the volcanic forcing data
    index_max = int(np.ceil(df_volcanic.index[-1])) - start + 1
    # Set volcanic forcing array values from the dataframe
    volcanic_forcing[:index_max] = df_volcanic.groupby(np.ceil(df_volcanic.index.values)//1).mean().squeeze().values
    # Volcanic forcing decreases to zero during the following 10 years after the last year with known volcanic forcing
    volcanic_forcing[(index_max-1):(index_max+9)] = np.linspace(1,0,10) * volcanic_forcing[index_max-1]
    # Solar forcing
    solar_forcing = df_solar["erf"].loc[start:end].values
    # Harmonized emissions
    da_emissions = xr.load_dataarray(
        f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}"
        "/emissions/ssp_emissions_1750-2500.nc")
    da = da_emissions.loc[dict(config="unspecified", scenario=scenario)][:(L-1), ...]
    fe = da.expand_dims(dim=["scenario","config"], axis=(1,2))
    emissions = fe.drop("config") * np.ones((1, nconfigs, 1))
    return solar_forcing, volcanic_forcing, emissions

def get_param_names():
    df = pd.read_csv(f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
                       "posteriors/calibrated_constrained_parameters.csv",index_col=0,nrows=0)
    df.rename({'co2_concentration_1750': 'CO2 concentration 1750'},axis='columns',inplace=True)
    return df.columns.tolist()

def load_calib_samples():
    df = pd.read_csv(f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
                     "posteriors/calibrated_constrained_parameters.csv",
                     index_col=0)
    df.rename({'co2_concentration_1750': 'CO2 concentration 1750'},axis='columns',inplace=True)
    return df

def constraint_ranges():
    out = {}
    out['ecs'] = (1.0,7.0)
    out['tcr'] = (0.5,3.5)
    out['T 1995-2014'] = (0.5,1.2)
    out['ari'] = (-1.0,0.4)
    out['aci'] = (-2.25, 0.1)
    out['aer'] = (-3, 0)
    out['CO2'] = (396, 400)
    out['ohc'] = (0, 800)
    out['T 2081-2100'] = (0.7, 3.2)
    return out

def get_param_ranges():
    return {'gamma': (0.0,25.0), 'c1': (2.0,7.0), 'c2':(0.0,50.0), 'c3':(0.0,250.0),
            'kappa1': (0.5,2.5), 'kappa2': (0.0,7.0), 'kappa3': (0.0,2.0), 'epsilon': (0.0,2.5),
            'sigma_eta': (0.0,2.5),'sigma_xi':(0.0,1.0),'F_4xCO2':(5.0,12.0),'r0':(20.0,45.0),
            'rU': (-0.01,0.02),'rT': (-2.5,7.5),'rA':(-0.005,0.008),'ari BC':(0.0,0.042),
            'ari CH4': (-4.54e-6,0.0),'ari N2O': (-7.27e-5,0.0), 'ari NH3': (-0.00113,0.0),
            'ari NOx': (-0.000135,0.0),'ari OC': (-0.00792,0.0), 'ari Sulfur': (-0.005725,0.0),
            'ari VOC': (-3.2e-5,0.0),'ari Equivalent effective stratospheric chlorine': (-1.5072e-5,0.0),
            'shape Sulfur': (0.0,0.05), 'shape BC': (0.0,0.7), 'shape OC': (0.0,0.05),'beta':(-3.0,0.0),
            'o3 CH4': (-7e-6,4e-4),'o3 N2O': (-1e-4,1.5e-3),
            'o3 Equivalent effective stratospheric chlorine': (-3.5e-4,1e-4),
            'o3 CO': (-1e-4,4e-4),'o3 VOC': (-4e-4,9e-4),'o3 NOx': (0.0,0.004), 'scale CH4': (0.5,1.5),
            'scale N2O': (0.7,1.3), 'scale minorGHG': (0.6,1.4), 'scale Stratospheric water vapour': (-0.5,2.5),
            'scale Contrails': (-0.2,2.2),'scale Light absorbing particles on snow and ice': (-0.6,2.6),
            'scale Land use': (0.0,2.0),'scale Volcanic': (0.5,1.5), 'solar_amplitude': (0.0,2.0), 
            'solar_trend': (-0.11,0.11),'scale CO2': (0.75,1.25), 'CO2 concentration 1750': (272,283), 
            'seed': (0,int(6e8))}
    
def constraint_priors():
    out = {}
    temp = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy")
    prior_ecs_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy")
    out['ecs'] = gaussian_kde(prior_ecs_samples)
    prior_tcr_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy")
    out['tcr'] = gaussian_kde(prior_tcr_samples)
    out['T 1995-2014'] = gaussian_kde(np.average(temp[145:166,:],axis=0) - np.average(temp[:52,:],axis=0))
    prior_ari_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ari_2005-2014_mean.npy")
    out['ari'] = gaussian_kde(prior_ari_samples)
    prior_aci_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_aci_2005-2014_mean.npy")
    out['aci'] = gaussian_kde(prior_aci_samples)
    prior_aer_samples = prior_aci_samples + prior_ari_samples
    out['aer'] = gaussian_kde(prior_aer_samples)
    prior_co2_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/concentration_co2_2014.npy")
    out['CO2'] = gaussian_kde(prior_co2_samples)
    prior_ohc_samples = np.load(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ocean_heat_content_2018_minus_1971.npy") / 1e21
    out['ohc'] = gaussian_kde(prior_ohc_samples)
    prior_future_samples = np.average(temp[231:252,:],axis=0) - np.average(temp[145:166,:],axis=0)
    out['T 2081-2100'] = gaussian_kde(prior_future_samples)
    return out

def constraint_targets():
    out = {}
    out['ecs'] = lambda x: skewnorm.pdf(x,8.82185594,loc=1.95059779,scale=1.55584604)
    out['tcr'] = lambda x: norm.pdf(x,loc=1.8, scale=0.6/norm.ppf(0.95))
    out['T 1995-2014'] = lambda x: skewnorm.pdf(x,-1.65506091,loc=0.92708099,scale=0.12096636)
    out['ari'] = lambda x: norm.pdf(x,loc=-0.3,scale=0.3/norm.ppf(0.95))
    out['aci'] = lambda x: norm.pdf(x,loc=-1.0,scale=0.7/norm.ppf(0.95))
    out['aer'] = lambda x: norm.pdf(x,loc=-1.3,scale=np.sqrt(0.7**2+0.3**2)/norm.ppf(0.95))
    out['CO2'] = lambda x: norm.pdf(x,loc=397.5469792683919,scale=0.36)
    out['ohc'] = lambda x: norm.pdf(x,loc=396/0.91,scale=67/0.91)
    out['T 2081-2100'] = lambda x: skewnorm.pdf(x,2.20496701,loc=1.4124379,scale=0.60080822)
    return out

def load_prior_samples():
    # Read and rename dataframes
    df_cr = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/climate_response_ebm3.csv")
    df_cc = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/carbon_cycle.csv")
    df_ari = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_radiation.csv")
    df_ari.drop(labels='CO',axis='columns',inplace=True)
    df_ari.rename({col: f'ari {col}' for col in df_ari.columns},axis='columns',inplace=True)
    df_aci = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_cloud.csv")
    df_aci.rename({col: f"shape {col.split('_')[-1].upper()}" for col in df_aci.columns if 'shape' in col},axis='columns',inplace=True)
    df_aci.rename({'shape SO2': "shape Sulfur"},axis='columns',inplace=True)
    df_o3 = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv")
    df_o3.rename({col: f"o3 {col}" for col in df_o3.columns},axis='columns',inplace=True)
    df_scaling = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/forcing_scaling.csv")
    df_scaling.rename({col: f'scale {col}' for col in df_scaling.columns if 'solar' not in col},axis='columns',inplace=True)
    df_co2 = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/co2_concentration_1750.csv")
    df_co2.rename({'co2_concentration': 'CO2 concentration 1750'},axis='columns',inplace=True)
    # Concatenate dataframes
    df = pd.concat((df_cr,df_cc,df_ari,df_aci,df_o3,df_scaling,df_co2),axis='columns')
    return df
    
def get_prior(included):
    param_ranges = get_param_ranges()
    # Kernel density estimation for climate response parameters
    df_cr = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/climate_response_ebm3.csv")
    for param in df_cr.columns:
        if param  not in included:
            df_cr.drop(labels=param,axis='columns',inplace=True)
    cr_d = gaussian_kde(df_cr.T) if len(df_cr.columns) != 0 else lambda x: 1
    
    # Kernel density estimation for carbon cycle parameters
    df_cc = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/carbon_cycle.csv")
    for param in df_cc.columns:
        if param not in included:
            df_cc.drop(labels=param,axis='columns',inplace=True)
    cc_d = gaussian_kde(df_cc.T) if len(df_cc.columns) != 0 else lambda x: 1
    
    # Aerosol-radiation interaction parameters
    df_ari = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_radiation.csv")
    df_ari.drop(labels='CO',axis='columns',inplace=True)
    df_ari.rename({col: f'ari {col}' for col in df_ari.columns},axis='columns',inplace=True)
    for param in df_ari.columns:
        if param  not in included:
            df_ari.drop(labels=param,axis='columns',inplace=True)
    if len(df_ari.columns) != 0:
        bounds = np.array([param_ranges[param] for param in df_ari.columns])
        ari_d = lambda x: np.prod(uniform.pdf(x,loc=bounds[:,0],scale=bounds[:,1]-bounds[:,0]))
    else:
        ari_d = lambda x: 1
    
    # Aerosol-cloud interaction parameters
    df_aci = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_cloud.csv")
    df_aci.rename({col: f"shape {col.split('_')[-1].upper()}" for col in df_aci.columns if 'shape' in col},axis='columns',inplace=True)
    df_aci.rename({'shape SO2': "shape Sulfur"},axis='columns',inplace=True)
    for param in df_aci.columns:
        if param not in included:
            df_aci.drop(labels=param,axis='columns',inplace=True)
    aci_d = gaussian_kde(df_aci.T) if len(df_aci.columns) != 0 else lambda x: 1
    
    # Ozone interaction parameters
    df_o3 = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv")
    df_o3.rename({col: f"o3 {col}" for col in df_o3.columns},axis='columns',inplace=True)
    for param in df_o3.columns:
        if param  not in included:
            df_o3.drop(labels=param,axis='columns',inplace=True)
    o3_d = gaussian_kde(df_o3.T) if len(df_o3.columns) != 0 else lambda x: 1
    
    # Scaling factor parameters
    df_scaling = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/forcing_scaling.csv")
    df_scaling.rename({col: f'scale {col}' for col in df_scaling.columns if 'solar' not in col},axis='columns',inplace=True)
    for param in df_scaling.columns:
        if param not in included:
            df_scaling.drop(labels=param,axis='columns',inplace=True)
    scaling_d = gaussian_kde(df_scaling.T) if len(df_scaling.columns) != 0 else lambda x: 1
    
    # CO2 concentration on year 1750
    df_co2 = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/co2_concentration_1750.csv")
    df_co2.rename({'co2_concentration': 'CO2 concentration 1750'},axis='columns',inplace=True)
    for param in df_co2.columns:
        if param not in included:
            df_co2.drop(labels=param,axis='columns',inplace=True)
    co2conc_d = gaussian_kde(df_co2.T) if len(df_co2.columns) != 0 else lambda x: 1
    N_params = [len(df_cr.columns),len(df_cc.columns),len(df_ari.columns),len(df_aci.columns),len(df_o3.columns),len(df_scaling.columns),len(df_co2.columns)]
    if sum(N_params) != len(included):
        raise ValueError(f'Number of parameters {sum(N_params)} in the prior mismatch the number {len(included)} of included parameters')
    
    prior = lambda x: float(cr_d(x[0:sum(N_params[:1])]) * cc_d(x[sum(N_params[:1]):sum(N_params[:2])]) * ari_d(x[sum(N_params[:2]):sum(N_params[:3])]) * \
                            aci_d(x[sum(N_params[:3]):sum(N_params[:4])]) * o3_d(x[sum(N_params[:4]):sum(N_params[:5])]) * \
                            scaling_d(x[sum(N_params[:5]):sum(N_params[:6])]) * co2conc_d(x[sum(N_params[:6]):sum(N_params[:7])]))
    return prior

def read_temperature_data():
    # Years
    #years = list(range(1850,2021))
    # Temperature data
    ar6_file = "AR6_GMST.csv"
    ar6_path = f"{fair_calibration_dir}/data/forcing/{ar6_file}"
    df = pd.read_csv(ar6_path, index_col=0).to_xarray()
    T_mean = df['gmst']
    # Std from HadCrut5 dataset
    hadcrut5_file = "HadCRUT.5.0.1.0.analysis.ensemble_series.global.monthly.nc"
    hadcrut5_path = f"{cwd}/leach-et-al-2021/data/input-data/Temperature-observations/{hadcrut5_file}"
    ds = xr.open_dataset(hadcrut5_path)
    T_data = ds['tas'].groupby('time.year').mean('time')
    # Change reference temperature to mean of 1851-1900
    #T_data = T_data - T_data.loc[dict(year=slice(1850,1900))].mean(dim='year')
    T_std = T_data.std(dim='realization')
    return T_mean, T_std

def resample_constraint_posteriors(N=10**5):
    NINETY_TO_ONESIGMA = norm.ppf(0.95)
    valid_temp = np.loadtxt(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
        "runids_rmse_pass.csv"
    ).astype(np.int64)

    input_ensemble_size = len(valid_temp)

    assert input_ensemble_size > output_ensemble_size

    temp_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "temperature_1850-2101.npy"
    )
    ohc_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "ocean_heat_content_2018_minus_1971.npy"
    )
    fari_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "forcing_ari_2005-2014_mean.npy"
    )
    faci_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "forcing_aci_2005-2014_mean.npy"
    )
    co2_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "concentration_co2_2014.npy"
    )
    ecs_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy"
    )
    tcr_in = np.load(
        f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy"
    )
    faer_in = fari_in + faci_in


    def opt(x, q05_desired, q50_desired, q95_desired):
        "x is (a, loc, scale) in that order."
        q05, q50, q95 = skewnorm.ppf((0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2])
        return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)
    
    constraints = ["ecs","tcr","ohc","T 1995-2014","ari","aci","aer","CO2","T 2081-2100"]
    
    samples = {}
    samples["ecs"] = skewnorm.rvs(8.82185594, loc=1.95059779, scale=1.55584604, size=N, random_state=91603)
    samples["tcr"] = norm.rvs(loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=N, random_state=18196)
    samples["ohc"] = norm.rvs(loc=396 / 0.91, scale=67 / 0.91, size=N, random_state=43178)
    samples["T 1995-2014"] = skewnorm.rvs(-1.65506091, loc=0.92708099, scale=0.12096636, size=N, random_state=19387)
    samples["ari"] = norm.rvs(loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=N, random_state=70173)
    samples["aci"] = norm.rvs(loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=N, random_state=91123)
    samples["aer"] = norm.rvs(loc=-1.3,scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_ONESIGMA,size=N,random_state=3916153)
    samples["CO2"] = norm.rvs(loc=397.5469792683919, scale=0.36, size=N, random_state=81693)
    samples["T 2081-2100"] = skewnorm.rvs(2.20496701, loc=1.4124379, scale=0.60080822, size=N, random_state=801693589)

    ar_distributions = {}
    for constraint in constraints:
        ar_distributions[constraint] = {}
        ar_distributions[constraint]["bins"] = np.histogram(
            samples[constraint], bins=100, density=True
        )[1]
        ar_distributions[constraint]["values"] = samples[constraint]

    weights_20yr = np.ones(21)
    weights_20yr[0] = 0.5
    weights_20yr[-1] = 0.5
    weights_51yr = np.ones(52)
    weights_51yr[0] = 0.5
    weights_51yr[-1] = 0.5

    accepted = pd.DataFrame(
        {
            "ecs": ecs_in[valid_temp],
            "tcr": tcr_in[valid_temp],
            "ohc": ohc_in[valid_temp] / 1e21,
            "T 1995-2014": np.average(temp_in[145:166, valid_temp], weights=weights_20yr, axis=0)
                         - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0),
            "ari": fari_in[valid_temp],
            "aci": faci_in[valid_temp],
            "aer": faer_in[valid_temp],
            "CO2": co2_in[valid_temp],
            "T 2081-2100": np.average(temp_in[231:252, valid_temp], weights=weights_20yr, axis=0)
                         - np.average(temp_in[145:166, valid_temp], weights=weights_20yr, axis=0)
        },
        index=valid_temp,
    )


    def calculate_sample_weights(distributions, samples, niterations=50):
        weights = np.ones(samples.shape[0])
        gofs = []
        gofs_full = []

        unique_codes = list(distributions.keys())  # [::-1]

        for k in range(niterations):
            gofs.append([])
            if k == (niterations - 1):
                weights_second_last_iteration = weights.copy()
                weights_to_average = []

            for j, unique_code in enumerate(unique_codes):
                unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                    unique_code, distributions, samples, weights, j, k
                )
                if k == (niterations - 1):
                    weights_to_average.append(unique_code_weights[our_values_bin_idx])

                weights *= unique_code_weights[our_values_bin_idx]

                gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
                gofs[-1].append(gof)

                gofs_full.append([unique_code])
                for unique_code_check in unique_codes:
                    unique_code_check_weights, _ = get_unique_code_weights(
                        unique_code_check, distributions, samples, weights, 1, 1
                    )
                    gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                    gofs_full[-1].append(gof)

        weights_stacked = np.vstack(weights_to_average).mean(axis=0)
        weights_final = weights_stacked * weights_second_last_iteration

        gofs_full.append(["Final iteration"])
        for unique_code_check in unique_codes:
            unique_code_check_weights, _ = get_unique_code_weights(
                unique_code_check, distributions, samples, weights_final, 1, 1
            )
            gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
            gofs_full[-1].append(gof)

        return (
            weights_final,
            pd.DataFrame(np.array(gofs), columns=unique_codes),
            pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
        )


    def get_unique_code_weights(unique_code, distributions, samples, weights, j, k):
        bin_edges = distributions[unique_code]["bins"]
        our_values = samples[unique_code].copy()

        our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
        np.testing.assert_allclose(bin_edges, bin_edges_np)
        assessed_ranges_bin_counts, _ = np.histogram(
            distributions[unique_code]["values"], bins=bin_edges
        )

        our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

        existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
        for i in range(existing_weighted_bin_counts.shape[0]):
            existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

        if np.equal(j, 0) and np.equal(k, 0):
            np.testing.assert_equal(
                existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
            )

        unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

        # existing_weighted_bin_counts[0] refers to samples outside the
        # assessed range's lower bound. Accordingly, if `our_values` was
        # digitized into a bin idx of zero, it should get a weight of zero.
        unique_code_weights[0] = 0
        # Similarly, if `our_values` was digitized into a bin idx greater
        # than the number of bins then it was outside the assessed range
        # so get a weight of zero.
        unique_code_weights[-1] = 0

        for i in range(1, our_values_bin_counts.shape[0] + 1):
            # the histogram idx is one less because digitize gives values in the
            # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
            histogram_idx = i - 1
            if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
                unique_code_weights[i] = 0
            elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
                # other variables force this box to be zero so just fill it with
                # one
                unique_code_weights[i] = 1
            else:
                unique_code_weights[i] = (
                    assessed_ranges_bin_counts[histogram_idx]
                    / existing_weighted_bin_counts[histogram_idx]
                )

        return unique_code_weights, our_values_bin_idx


    weights, gofs, gofs_full = calculate_sample_weights(
        ar_distributions, accepted, niterations=30
    )

    effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
    print("Number of effective samples:", effective_samples)

    assert effective_samples >= output_ensemble_size

    drawn_samples = accepted.sample(n=output_ensemble_size,replace=False,weights=weights,random_state=10099)
    drawn_samples.reset_index(inplace=True)
    return drawn_samples
    
def load_MC_samples(ds,N=None,tail=True,thinning=1,param_ranges=None):
    #samples = ds['sample'].size
    names = ds['param'][:]
    warmup = ds['warmup'][:]
    chain = ds['chain'][warmup:,:]
    seeds = ds['seeds'][warmup:]
    configs = pd.DataFrame(columns=names)
    excluded = [param for param in param_ranges.keys() if param not in names]
    total_samples = len(chain)
    if N is None:
        N = total_samples // thinning
        required_samples = total_samples * thinning
    else:
        required_samples = N * thinning
        if required_samples > total_samples:
            raise ValueError(f'Number of required samples {required_samples} is larger than the number of total samples {total_samples}')
    if tail:
        configs[names] = chain[-required_samples::thinning,:]
        configs['seed'] = seeds[-required_samples::thinning]
    else:
        configs[names] = chain[:required_samples,:][::thinning]
        configs['seed'] = seeds[:required_samples][::thinning]
    for param in excluded:
        if param == 'seed':
            continue
        configs[param] = npr.uniform(min(param_ranges[param]),max(param_ranges[param]),size=N)
    return configs

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
    fill(f.climate_configs["ocean_heat_transfer"],df_configs.loc[:,"kappa1":"kappa3"].values)
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
    #f.fill_from_rcmip()
    
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
        df_configs["CO2 concentration 1750"].values.squeeze(),
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

def compute_data_loss(model,data,var):
    wss = np.sum(np.square(model-data)/var)
    loss = 0.5 * (wss + np.sum(np.log(2*np.pi*var)))
    return loss

def compute_constrained_loss(constraints,targets,weights=None):
    names = ['ecs','tcr','T 1995-2014','ari','aci','aer', 'CO2', 'ohc', 'T 2081-2100']
    densities = np.array([targets[constraint](constraints[i]) for i, constraint in enumerate(names)])
    if weights is None:
        weights = np.ones(len(names))
    return -np.sum(np.log(densities) * weights) if np.all(densities) != 0.0 else np.inf

def compute_prior_loss(prior_fun,x):
    prior_density_value = prior_fun(x)
    return -np.log(prior_density_value) if prior_density_value != 0.0 else np.inf

def compute_constraints(fair):
    out = np.full(9,np.nan)
    # Equilibrium climate sensitivity
    out[0] = float(fair.ebms.ecs.to_numpy())
    # Transient climate response
    out[1] = float(fair.ebms.tcr.to_numpy())
    # Average temperarature between years 1995-2014 referenced with temperature between years 1850-1901   
    out[2] = float(fair.temperature.loc[dict(timebounds=slice(1996,2015),scenario='ssp245',layer=0)].mean(dim='timebounds').to_numpy() -
                   fair.temperature.loc[dict(timebounds=slice(1851,1901),scenario='ssp245',layer=0)].mean(dim='timebounds').to_numpy())
    # Average aerosol-radiation interactions between 2005-2014 compared to the year 1750
    out[3] = float(fair.forcing.loc[dict(timebounds=slice(2006,2015),scenario='ssp245',specie='Aerosol-radiation interactions')].mean(dim='timebounds').to_numpy() -
                   fair.forcing.loc[dict(timebounds=1750,scenario='ssp245',specie='Aerosol-radiation interactions')].to_numpy().squeeze())
    # Average aerosol-cloud interactions between 2005-2014
    out[4] = float(fair.forcing.loc[dict(timebounds=slice(2006,2015),scenario='ssp245',specie='Aerosol-cloud interactions')].mean(dim='timebounds').to_numpy() - 
                   fair.forcing.loc[dict(timebounds=1750,scenario='ssp245',specie='Aerosol-cloud interactions')].to_numpy().squeeze())
    # Total aerosol interaction
    out[5] = out[3] + out[4]
    # CO2 concentration in year 2014
    out[6] = float(fair.concentration.loc[dict(timebounds=2015,scenario='ssp245',specie='CO2')].to_numpy().squeeze())
    # Ocean heat content change between a year 1971 and a year 2018
    out[7] = float(fair.ocean_heat_content_change.loc[dict(timebounds=2019,scenario='ssp245')].to_numpy() -
                   fair.ocean_heat_content_change.loc[dict(timebounds=1972,scenario='ssp245')].to_numpy()) / 1e21
    # Average temperature between years 2081-2100 compared with temperature from years 1995-2014 using scenario ssp245
    out[8] = float(fair.temperature.loc[dict(timebounds=slice(2082,2101),scenario='ssp245',layer=0)].mean(dim='timebounds').to_numpy() -
                   fair.temperature.loc[dict(timebounds=slice(1996,2015),scenario='ssp245',layer=0)].mean(dim='timebounds').to_numpy())
    return out
