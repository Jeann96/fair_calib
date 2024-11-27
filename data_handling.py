#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:04:47 2024

@author: nurmelaj
"""

import os
cwd = os.getcwd()
from pandas import read_csv,concat
import numpy as np
from netCDF4 import Dataset
from xarray import load_dataarray, load_dataset
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
cal_v = out["CALIBRATION_VERSION"]
fair_v = out["FAIR_VERSION"]
constraint_set = out["CONSTRAINT_SET"]
prior_samples = int(out["PRIOR_SAMPLES"])
output_ensemble_size = int(out["POSTERIOR_SAMPLES"])
make_plots = True if out['PLOTS'] == 'True' else False
datadir = f"{cwd}/{out['DATADIR']}"
samplingdir = f"{cwd}/{out['SAMPLINGDIR']}"
figdir = f"{cwd}/{out['FIGDIR']}"

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
    required_variables = ['filename','warmup','samples','thinning','parallel_chains','stochastic',
                          'scenario','use_priors','use_constraints','use_Tvar_constraint']
    for variable in required_variables:
        if variable not in out.keys():
            raise ValueError(f'Variable {variable} missing from the settings file {os.path.basename(path)}')
    for variable in out.keys():    
        if variable not in out.keys():
            raise ValueError(f'Extra variable {variable} in the settings file {os.path.basename(path)}')
    return out        

def read_calib_samples(dtype=np.float64):
    df = read_csv(f"{datadir}/output/fair-{fair_v}/v{cal_v}/all-2022/posteriors/calibrated_constrained_parameters.csv",
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
    df_volcanic = read_csv(f"{datadir}/data/forcing/volcanic_ERF_1750-2101_timebounds.csv",
                              index_col='timebounds')
    df_solar = read_csv(f"{datadir}/data/forcing/solar_erf_timebounds.csv",
                           index_col="year")
    volcanic_forcing = np.zeros(end-start+1,dtype=dtype)
    solar_forcing = np.zeros(end-start+1,dtype=dtype)
    volcanic_forcing[0:(min(2101,end)-start+1)] = df_volcanic["erf"].loc[start:end].values
    solar_forcing[0:(min(2300,end)-start+1)] = df_solar["erf"].loc[start:end].values
    return solar_forcing, volcanic_forcing

def read_emissions(start,end,scenarios,nconfigs=1,folder=None,dtype=np.float64):
    if folder is None:
        folder = f"{datadir}/output/fair-{fair_v}/v{cal_v}/all-2022/emissions"
    # Harmonized emissions 
    da_emissions = load_dataarray(f"{folder}/ssps_harmonized_1750-2499.nc").astype(dtype)
    da_emissions = da_emissions.sel(timepoints=slice(start,end),scenario=scenarios,config=["unspecified"])
    return da_emissions.drop('config') * np.ones((1, 1, nconfigs, 1), dtype=dtype)

def read_prior_samples(folder_path=None,dtype=np.float64,validated=False):
    if folder_path is None:
        folder_path = f"{datadir}/output/fair-{fair_v}/v{cal_v}/all-2022/priors"
    if validated:
        df = read_csv(f"{folder_path}/prior_samples_validated.csv").astype(dtype)
    else:
        df_cr = read_csv(f"{folder_path}/climate_response_ebm3.csv").astype(dtype)
        df_cc = read_csv(f"{folder_path}/carbon_cycle.csv").astype(dtype)
        df_ari = read_csv(f"{folder_path}/aerosol_radiation.csv").astype(dtype)
        df_aci = read_csv(f"{folder_path}/aerosol_cloud.csv").astype(dtype)
        df_o3 = read_csv(f"{folder_path}/ozone.csv").astype(dtype)
        df_scaling = read_csv(f"{folder_path}/forcing_scaling.csv").astype(dtype)
        df_co2 = read_csv(f"{folder_path}/co2_concentration_1750.csv").astype(dtype)
        # Rename some of the columns
        df_cr.rename({col: f'clim_{col}' for col in df_cr.columns},axis='columns',inplace=True)
        df_cc.rename({col: f'cc_{col}' for col in df_cc.columns},axis='columns',inplace=True)
        df_ari.rename({'Equivalent effective stratospheric chlorine':'Ee_stratospheric_Cl'},axis='columns',inplace=True)
        df_ari.rename({col: f'ari_{col}' for col in df_ari.columns},axis='columns',inplace=True)
        df_aci.rename({'shape_so2':'shape_SO2','shape_oc':'shape_OC','shape_bc':'shape_BC'},axis='columns',inplace=True)
        df_aci.rename({col: f'aci_{col}' for col in df_aci.columns},axis='columns',inplace=True)
        df_o3.rename({'Equivalent effective stratospheric chlorine': "Ee_stratospheric_Cl"},axis='columns',inplace=True)
        df_o3.rename({col: f"o3_{col}" for col in df_o3.columns},axis='columns',inplace=True)
        df_scaling.rename({'Light absorbing particles on snow and ice': 'light_abs_particles',
                           'Stratospheric water vapour':'stratospheric_H2O_vapor'},
                          axis='columns',inplace=True)
        df_scaling.rename({col: f'fscale_{col}' for col in df_scaling.columns},axis='columns',inplace=True)
        df_co2.rename({'co2_concentration': 'cc_CO2_conc_1750'},axis='columns',inplace=True)
        df = concat((df_cr,df_cc,df_ari,df_aci,df_o3,df_scaling,df_co2),axis='columns')
    return df

def get_param_names():
    df = read_csv(f"{datadir}/output/fair-{fair_v}/v{cal_v}/all-2022/posteriors/calibrated_constrained_parameters.csv",
                     index_col=0,nrows=0)
    df.rename({'co2_concentration_1750': 'CO2 concentration 1750'},axis='columns',inplace=True)
    return df.columns.tolist()
    
def read_temperature_obs(gmst_obs=True,return_unc=True,round_fractional_year=True,
                         realizations=False,start_year=1850,end_year=2023):
    if gmst_obs:
        if realizations:
            print('Option "realization=True" is being ignored for gmst data')
        # Global mean surface temperature (gmst) observations between 1850 and 2023
        ar6_path = f"{datadir}/data/forcing/IGCC_GMST_1850-2022.csv"
        obs = read_csv(ar6_path, index_col=0)['gmst'].to_xarray()
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
    if return_unc:
        return obs,unc
    else:
        return obs

def create_file(scenario,paraller_chains,params,constraints,warmup,filename='sampling'):
    # Create nc-file for writing
    os.makedirs(samplingdir,exist_ok=True)
    ncfile = Dataset(f'{samplingdir}/{filename}.nc',mode='w')
    ncfile.createDimension('sample', None)
    ncfile.createVariable('sample', np.uint32, ['sample'], fill_value=False)
    ncfile.createDimension('param', len(params))
    ncfile.createVariable('param', str, ['param'], fill_value=False)
    ncfile['param'][:] = np.array(params,dtype=str)
    ncfile.createDimension('chain_id', paraller_chains)
    ncfile.createVariable('chain_id', np.uint32, ['chain_id'], fill_value=False)
    ncfile['chain_id'][:] = np.arange(paraller_chains,dtype=np.uint32)
    ncfile.createDimension('constraint', len(constraints))
    ncfile.createVariable('constraint', str, ['constraint'], fill_value=False)
    ncfile['constraint'][:] = np.array(constraints,dtype=str)
    ncfile.title = 'FaIR MC sampling'
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

def read_jumping_cov(params):
    cov_path = f'{datadir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/jumping_cov.csv'
    if os.path.exists(cov_path):
        return read_csv(cov_path,index_col=0).to_numpy()
    else:
        log_scale_transformation = {'aci_beta': lambda x: np.log(-x),
                                    'aci_shape_SO2': lambda x: np.log(x),
                                    'aci_shape_BC': lambda x: np.log(x),
                                    'aci_shape_OC': lambda x: np.log(x)}
        name_changes = {'aci_beta': 'log(-aci_beta)',
                        'aci_shape_SO2': 'log(aci_shape_SO2)',
                        'aci_shape_BC': 'log(aci_shape_BC)',
                        'aci_shape_OC': 'log(aci_shape_OC)'}
        prior_configs = transform_df(read_prior_samples(validated=True),log_scale_transformation,name_changes=name_changes)
        return np.diag((prior_configs[params].std(axis='rows',ddof=1).to_numpy() / 2.0)**2)
        
        