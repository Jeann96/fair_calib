#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:41:59 2023
@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.stats import norm
from dotenv import load_dotenv
from fair_tools import load_configs, load_MC_configs, run_1pctco2, runFaIR


load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")


cwd = os.getcwd()

figdir=f"{cwd}/figures"

fair_calibration_dir=f"{cwd}/fair-calibrate"


df_configs = load_configs()


#Make normal distribution for TCRE (K / 1000 Gt C)
tcre_50=1.65
tcre_83=2.3

z = norm.ppf((1 + 0.66) / 2)
sd = (tcre_83 - tcre_50) / z
tcre_norm=norm(loc=tcre_50, scale=sd)

#Make normal distribution for TCR (K / 1000 Gt C)
tcr_50=1.8
tcr_83=2.2

z = norm.ppf((1 + 0.66) / 2)
sd = (tcr_83 - tcr_50) / z
tcr_norm=norm(loc=tcr_50, scale=sd)


scenario = 'ssp245'
ignored = ['ari BC', 'ari CH4', 'ari N2O', 'ari NH3', 'ari NOx', 'ari OC', 'ari Sulfur', 'ari VOC',
           'ari Equivalent effective stratospheric chlorine','seed']
included = [param for param in df_configs.columns if param not in ignored]
sampled_configs = load_MC_configs('MC_results/',scenario,included)

tcre, tcr=run_1pctco2(df_configs)
tcre_mcmc, tcr_mcmc=run_1pctco2(sampled_configs)

fig, ax=pl.subplots(1,2)
tcre.plot.hist(bins=100,ax=ax[0], density=True, label='Smith et al.')
tcre_mcmc.plot.hist(bins=100,ax=ax[0], density=True, label='MCMC')
tcre_grid=np.linspace(0.2,3.5)
ax[0].plot(tcre_grid,tcre_norm.pdf(tcre_grid), label='IPCC AR6')
ax[0].set_title('TCRE')
ax[0].set_xlabel('K / 1000 Gt C')
ax[0].set_xlim([0,4])
ax[0].legend()

tcr.plot.hist(bins=100,ax=ax[1], density=True, label='Smith et al.')
tcr_mcmc.plot.hist(bins=100,ax=ax[1], density=True, label='MCMC',range=(0,3))
tcr_grid=np.linspace(0.2,5)
ax[1].plot(tcr_grid,tcr_norm.pdf(tcr_grid), label='IPCC AR6')
ax[1].set_title('TCR')
ax[1].set_xlabel('K')
ax[1].set_xlim([0,4])
ax[1].legend()

fig.savefig(f'{figdir}/TCRE_and_TCR_pdfs.png', dpi=150)


