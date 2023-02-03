#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:18:27 2023

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as pl

cwd = os.getcwd()

figdir=f"{cwd}/figures"
datadir=f"{cwd}/data"

fair_calibration_dir=f"{cwd}/fair-calibrate"
fair_2_dir=f"{cwd}/leach-et-al-2021"

temp_datadir=f"{fair_2_dir}/aux/input-data/Temperature-observations"

data=dict()
mean=dict()
std=dict()
# data['HadCrut5']=xr.open_dataset(f'{datadir}/HadCRUT.5.0.1.0.analysis.summary_series.global.annual.nc')
data['HadCrut5-m']=xr.open_dataset(f'{temp_datadir}/HadCRUT.5.0.1.0.analysis.ensemble_series.global.monthly.nc')
data['HadCrut5']=data['HadCrut5-m'].tas.groupby('time.year').mean('time')

# Change reference temperature to mean of 1851-1900
data['HadCrut5']=data['HadCrut5']-data['HadCrut5'].loc[dict(year=slice(1851,1901))].mean(dim='year')

mean['HadCrut5']=data['HadCrut5'].mean(dim='realization')
std['HadCrut5']=data['HadCrut5'].std(dim='realization')
fig1, ax1=pl.subplots(1,1)

data['HadCrut5'].plot.line(x='year', ax=ax1, color='gray')
mean['HadCrut5'].plot(color='black', ax=ax1)
(mean['HadCrut5']+std['HadCrut5']).plot(color='black', linestyle='--',ax=ax1)
(mean['HadCrut5']-std['HadCrut5']).plot(color='black', linestyle='--',ax=ax1)

ax1.get_legend().remove()
fig1.savefig(f'{figdir}/HADCRUT5.png', dpi=150)

mean['HadCrut5'].to_netcdf(f'{datadir}/HadCrut5_mean.nc')
std['HadCrut5'].to_netcdf(f'{datadir}/HadCrut5_std.nc')
