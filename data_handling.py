#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:04:47 2024

@author: nurmelaj
"""

import os
cwd = os.getcwd()
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from netCDF4 import Dataset
from xarray import load_dataarray, load_dataset, DataArray
from datetime import datetime

def load_dotenv():
    out = {}
    with open('.env','r') as envfile:
        for line in envfile:
            if '=' not in line:
                continue
            key, val = line.strip().split('=')
            out[key] = val
    return out

out = load_dotenv()
fair_calib_dir = f"{cwd}/fair-calibrate"
mc_sampling_dir = f'{cwd}/MC_sampling'
cal_v = out["CALIBRATION_VERSION"]
fair_v = out["FAIR_VERSION"]
constraint_set = out["CONSTRAINT_SET"]
prior_samples = int(out["PRIOR_SAMPLES"])
output_ensemble_size = int(out["POSTERIOR_SAMPLES"])

def read_settings(path):
    out = {}
    with open(path) as file:
        for line in file:
            if line.count('=') != 1:
                continue
            line = line.strip().replace(' ', '')
            key, val = line.split('=')
            if val.isnumeric():
                out[key] = int(val)
            elif val in ['True','False']:
                out[key] = True if val == 'True' else False
            else:
                if '"' in val:
                    val = val.strip('"')
                elif "'" in val:
                    val = val.strip("'")
                out[key] = val
    required_variables = ['warmup','samples','thinning','scenario','use_default_priors','use_default_constraints',
                          'T_variability_constraint','target_log_likelihood','stochastic','plot']
    for variable in required_variables:
        if variable not in out.keys():
            raise ValueError(f'Variable {variable} missing from the settings file {os.path.basename(path)}')
    for variable in out.keys():    
        if variable not in out.keys():
            raise ValueError(f'Extra variable {variable} in the settings file {os.path.basename(path)}')
    return out

def read_calib_samples(dtype=np.float64):
    df = pd.read_csv(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/all-2022/posteriors/calibrated_constrained_parameters.csv",
                     index_col=0)
    df = df.astype({col: dtype for col in df.columns if col != 'seed'})
    df.rename({'cc_co2_concentration_1750':'cc_CO2_conc_1750',
               'ari_Equivalent effective stratospheric chlorine':'ari_Ee_stratospheric_Cl',
               'aci_shape_so2':'aci_shape_SO2','aci_shape_bc':'aci_shape_BC','aci_shape_oc':'aci_shape_OC',
               'o3_Equivalent effective stratospheric chlorine':'o3_Ee_stratospheric_Cl',
               'fscale_Stratospheric water vapour':'fscale_stratospheric_H2O_vapor',
               'fscale_Light absorbing particles on snow and ice':'fscale_light_abs_particles'},
               axis='columns',inplace=True)
    # Convert
    return df

def read_forcing(start,end,dtype=np.float64):
    # Volcanic and solar forcing
    df_volcanic = pd.read_csv(f"{fair_calib_dir}/data/forcing/volcanic_ERF_1750-2101_timebounds.csv",
                              index_col='timebounds')
    df_solar = pd.read_csv(f"{fair_calib_dir}/data/forcing/solar_erf_timebounds.csv",
                           index_col="year")
    volcanic_forcing = np.zeros(end-start+1,dtype=dtype)
    solar_forcing = np.zeros(end-start+1,dtype=dtype)
    volcanic_forcing[0:(min(2101,end)-start+1)] = df_volcanic["erf"].loc[start:end].values
    solar_forcing[0:(min(2300,end)-start+1)] = df_solar["erf"].loc[start:end].values
    return solar_forcing, volcanic_forcing

def read_emissions(start,end,scenarios,nconfigs=1,folder=None,dtype=np.float64):
    if folder is None:
        folder = f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/all-2022/emissions"
    # Harmonized emissions 
    da_emissions = load_dataarray(f"{folder}/ssps_harmonized_1750-2499.nc").astype(dtype)
    da_emissions = da_emissions.sel(timepoints=slice(start,end),scenario=scenarios,config=["unspecified"])
    return da_emissions.drop('config') * np.ones((1, 1, nconfigs, 1), dtype=dtype)

def read_prior_samples(folder_path=None,dtype=np.float64):
    if folder_path is None:
        folder_path = f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/all-2022/priors"
    df_cr = pd.read_csv(f"{folder_path}/climate_response_ebm3.csv").astype(dtype)
    df_cr.rename({col: f'clim_{col}' for col in df_cr.columns},axis='columns',inplace=True)
    df_cc = pd.read_csv(f"{folder_path}/carbon_cycle.csv").astype(dtype)
    df_cc.rename({col: f'cc_{col}' for col in df_cc.columns},axis='columns',inplace=True)
    df_ari = pd.read_csv(f"{folder_path}/aerosol_radiation.csv").astype(dtype)
    #df_ari.drop(labels='CO',axis='columns',inplace=True)
    df_ari.rename({'Equivalent effective stratospheric chlorine':'Ee_stratospheric_Cl'},axis='columns',inplace=True)
    df_ari.rename({col: f'ari_{col}' for col in df_ari.columns},axis='columns',inplace=True)
    df_aci = pd.read_csv(f"{folder_path}/aerosol_cloud.csv").astype(dtype)
    #df_aci.rename({'shape_so2': 'shape_SO2','shape_bc':'shape_BC','shape_oc':'shape_OC'},axis='columns',inplace=True)
    df_aci.rename({col: f'aci_{col}' for col in df_aci.columns},axis='columns',inplace=True)
    df_o3 = pd.read_csv(f"{folder_path}/ozone.csv").astype(dtype)
    df_o3.rename({'Equivalent effective stratospheric chlorine': "Ee_stratospheric_Cl"},axis='columns',inplace=True)
    df_o3.rename({col: f"o3_{col}" for col in df_o3.columns},axis='columns',inplace=True)
    df_scaling = pd.read_csv(f"{folder_path}/forcing_scaling.csv").astype(dtype)
    df_scaling.rename({'Light absorbing particles on snow and ice': 'light_abs_particles','Stratospheric water vapour':'stratospheric_H2O_vapor'},
                      axis='columns',inplace=True)
    df_scaling.rename({col: f'fscale_{col}' for col in df_scaling.columns},axis='columns',inplace=True)
    df_co2 = pd.read_csv(f"{folder_path}/co2_concentration_1750.csv").astype(dtype)
    df_co2.rename({'co2_concentration': 'cc_CO2_conc_1750'},axis='columns',inplace=True)
    # Concatenate dataframes
    df = pd.concat((df_cr,df_cc,df_ari,df_aci,df_o3,df_scaling,df_co2),axis='columns')
    return df

def get_param_names():
    df = pd.read_csv(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/all-2022/posteriors/calibrated_constrained_parameters.csv",
                     index_col=0,nrows=0)
    df.rename({'co2_concentration_1750': 'CO2 concentration 1750'},axis='columns',inplace=True)
    return df.columns.tolist()
    
def read_temperature_obs(gmst_obs=True,return_unc=True,round_fractional_year=True,
                         realizations=False,start_year=1850,end_year=2023):
    if gmst_obs:
        if realizations:
            print('Option "realization=True" is being ignored for gmst data')
        # Global mean surface temperature (gmst) observations between 1850 and 2023
        ar6_path = f"{fair_calib_dir}/data/forcing/IGCC_GMST_1850-2022.csv"
        obs = pd.read_csv(ar6_path, index_col=0)['gmst'].to_xarray()
    # Read HadCRUT data
    hadcrut5_file = "HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.nc"
    hadcrut5_path = f"{cwd}/leach-et-al-2021/data/input-data/Temperature-observations/{hadcrut5_file}"
    ds = load_dataset(hadcrut5_path)
    T = ds['tas']
    if not gmst_obs:
        obs = T if realizations else T.mean(dim='realization')
        obs = obs.drop_vars(['latitude','longitude'])
    # Uncertainty from HadCRUT realizations
    if return_unc:
        unc = np.sqrt(ds['coverage_unc']**2 + T.std(dim='realization',ddof=1)**2)
        unc = unc.drop_vars(['latitude','longitude'])
    timestamps = ds['time'].data
    ds.close()
    
    # If returning either HadCRUT mean as observation or HadCRUT uncertainty, 
    # datetime coordinates must be converted to fractional year
    if not gmst_obs or return_unc:
        # Function to convert timestamp to format {year}.{fractional_year}
        def numpydatetime_to_datetime(dt64):
            return datetime.utcfromtimestamp((dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s'))
        timestamps = list(map(numpydatetime_to_datetime,timestamps))
        def conversion(dt):
            year = dt.year
            delta = dt - datetime(year,1,1,0,0,0)
            return year + (delta.days + delta.seconds/(24*3600)) / 365.2425
        timestamps = list(map(conversion,timestamps))
        new_coords = dict(zip(['time'] + [dim for dim in T.dims if dim != 'time'],
                              [timestamps] + [T[dim].data for dim in T.dims if dim != 'time']))
        if not gmst_obs:
            obs = obs.assign_coords(new_coords) if realizations else obs.assign_coords({'time': timestamps})
            if round_fractional_year:
                obs['time'] = 0.5*np.round(2*obs['time'].data)
        if return_unc:
            unc = unc.assign_coords(new_coords) if realizations else unc.assign_coords({'time': timestamps})
            if round_fractional_year:
                unc['time'] = 0.5*np.round(2*unc['time'].data)
    # Drop coordinates outiside the specified year range
    obs = obs.where((start_year < obs['time']) & (obs['time'] < end_year), drop=True)
    if return_unc:
        unc = unc.where((start_year < unc['time']) & (unc['time'] < end_year), drop=True)
    '''
    if match_with_gmst:
        gmst = read_gmst_temperature()
        bool_arr = (np.floor(np.min(gmst['time'])) < T['time']) & (T['time'] < np.ceil(np.max(gmst['time'])))
        T_to_match = T.mean(dim='realization').where(bool_arr,drop=True)
        T0 = np.mean(T_to_match.data - gmst.data)
        T_shift = least_squares(lambda T_shift: T_to_match.data + T_shift - gmst.data,T0).x
        T = T + DataArray(data=T_shift*np.ones(T['time'].size),dims=['time'],coords=dict(timebounds=('time',T['time'].data)))
    '''
    if return_unc:
        return obs,unc
    else:
        return obs
    

def read_MC_samples(ds,param_ranges,param_means,param_stds,N=None,tail=True,thinning=1):
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
        if param == 'seed' or param.startswith('ari'):
            configs[param] = npr.uniform(min(param_ranges[param]),max(param_ranges[param]),size=N)
        elif param in ['clim_gamma','clim_sigma_eta','clim_sigma_xi']:
            shape = param_means[param]**2 / param_stds[param]**2
            scale = param_stds[param]**2 / param_means[param]
            configs[param] = npr.gamma(shape,scale,size=N)
        else:
            raise ValueError(f'Distribution for the excluded parameter {param} missing')
    return configs

def read_chain(path):
    return Dataset(path,mode='r')

def create_file(scenario,paraller_chains,params,warmup,filename='sampling'):
    # Create nc-file for writing
    ncfile = Dataset(f'MC_results/{filename}.nc',mode='w')
    ncfile.createDimension('sample', None)
    ncfile.createVariable('sample', np.uint32, ['sample'], fill_value=False)
    ncfile.createDimension('param', len(params))
    ncfile.createVariable('param', str, ['param'], fill_value=False)
    ncfile['param'][:] = np.array(params,dtype=str)
    ncfile.createDimension('chain_id', paraller_chains)
    ncfile.createVariable('chain_id', np.uint32, ['chain_id'], fill_value=False)
    ncfile['chain_id'][:] = np.arange(paraller_chains,dtype=np.uint32)
    constraints = ["ECS","TCR","Tinc","ERFari","ERFaci","ERFaer","CO2conc2022","OHC","Tvar"]
    ncfile.createDimension('constraint', len(constraints))
    ncfile.createVariable('constraint', str, ['constraint'], fill_value=False)
    ncfile['constraint'][:] = np.array(constraints,dtype=str)
    ncfile.title = 'FaIR MC sampling'
    #ncfile.scenario = scenario
    # Variables with dimensions
    ncfile.createVariable('chain',float,['sample','chain_id','param'],fill_value=False)
    ncfile.createVariable('loss',float,['sample','chain_id'],fill_value=False)
    ncfile.createVariable('seed',np.uint32,['sample','chain_id'],fill_value=False)
    ncfile.createVariable('constraints',float,['sample','chain_id','constraint'],fill_value=np.nan)
    # Variables without dimensions
    ncfile.createVariable('warmup',np.uint32,fill_value=False)
    ncfile['warmup'][:] = warmup
    ncfile.scenario = scenario
    # Close nc file
    ncfile.close()

def save_progress(chain_xr,loss_xr,seed_xr,constraint_xr,
                  folder='MC_results',filename='sampling'):
    N = chain_xr['sample'].size
    # Open nc-file for appending
    ncfile = Dataset(f'{folder}/{filename}.nc',mode='a')
    
    index = ncfile['sample'][:]
    if len(index) == 0:
        ncfile['sample'][:] = np.arange(N,dtype=np.uint32)
        ncfile['chain'][0:N] = chain_xr.data
        ncfile['loss'][0:N] = loss_xr.data
        ncfile['seed'][0:N] = seed_xr.data.astype(np.uint32)
        ncfile['constraints'][0:N] = constraint_xr.data
    else:
        ncfile['sample'][:] = np.append(index,np.arange(index[-1]+1,N,dtype=np.uint32))
        ncfile['chain'][(index[-1]+1):N] = chain_xr[(index[-1]+1):].data
        ncfile['loss'][(index[-1]+1):N] = loss_xr[(index[-1]+1):].data
        ncfile['seed'][(index[-1]+1):N] = seed_xr[(index[-1]+1):].data.astype(np.uint32)
        ncfile['constraints'][(index[-1]+1):N] = constraint_xr[(index[-1]+1):]
    # Close nc file
    ncfile.close()
    
def transform_df(df,transformation_dict,name_changes=None):
    if transformation_dict is None:
        return df
    columns = df.columns
    mapper = {col:transformation_dict[col] if col in transformation_dict.keys() else lambda x: x for col in columns}
    df = df.transform(mapper,axis='index')
    if name_changes is not None:
        df.rename(name_changes,axis='columns',inplace=True)
    return df