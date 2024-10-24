#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 2023

@author: nurmelaj
"""


import numpy as np
from pandas import read_csv
import xarray as xr
import warnings
from scipy.stats import chi2,norm,skewnorm,gaussian_kde
from scipy.optimize import root
from scipy.linalg import expm
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from data_handling import cwd,fair_v,cal_v,constraint_set,output_ensemble_size,fair_calib_dir
from data_handling import read_emissions,read_temperature_obs,transform_df
from pandas import DataFrame
from fair.energy_balance_model import EnergyBalanceModel

def validate_config(configs,bounds_dict=None,stochastic_run=False):
    #These parameters require postivity and other constraints
    climate_response_params = ['clim_gamma', 'clim_c1', 'clim_c2', 'clim_c3', 'clim_kappa1', 'clim_kappa2', 'clim_kappa3', 
                               'clim_epsilon', 'clim_sigma_eta', 'clim_sigma_xi','clim_F_4xCO2']
    # Climate response parameters must be positive
    positive = (configs[climate_response_params] >= 0.0).all(axis=1).to_numpy()
    #print(positive,end=' ')
    # Other conditions for parameter to be physical
    physical = ((configs['clim_gamma'] > 0.5) & 
                (configs['clim_c1'] > 2.0) & 
                (configs['clim_c2'] > configs['clim_c1']) & 
                (configs['clim_c3'] > configs['clim_c2']) & 
                (configs['clim_kappa1'] > 0.3)).to_numpy()
    # Strict bounds for the variables
    if bounds_dict is not None:
        params = list(configs.columns)
        min_lims, max_lims = np.transpose([bounds_dict[param] for param in params])
        arr = configs[params].to_numpy()
        inside_bounds = ((min_lims[np.newaxis,:] < arr) & (arr < max_lims[np.newaxis,:])).all(axis=1)
    if stochastic_run:
        stochasticity_ok = np.zeros(len(configs),dtype=bool)
        for n in range(len(configs)):
            ohc_params = configs.iloc[n, [1, 2, 3]]
            oht_params = configs.iloc[n, [4, 5, 6]]
            ebm = EnergyBalanceModel(ocean_heat_capacity=ohc_params,
                                     ocean_heat_transfer=oht_params,
                                     deep_ocean_efficacy=configs.iloc[n, 7],
                                     gamma_autocorrelation=configs.iloc[n, 0],
                                     sigma_xi=configs.iloc[n, 9],
                                     sigma_eta=configs.iloc[n, 8],
                                     forcing_4co2=configs.iloc[n, 10],
                                     stochastic_run=stochastic_run)
            eb_matrix = ebm._eb_matrix()
            q_mat = np.zeros((4, 4))
            q_mat[0, 0] = ebm.sigma_eta**2
            q_mat[1, 1] = (ebm.sigma_xi / ebm.ocean_heat_capacity[0]) ** 2
            h_mat = np.zeros((8, 8))
            h_mat[:4, :4] = -eb_matrix
            h_mat[:4, 4:] = q_mat
            h_mat[4:, 4:] = eb_matrix.T
            g_mat = expm(h_mat)
            q_mat_d = g_mat[4:, 4:].T @ g_mat[:4, 4:]
            # Check if the q_mat_d matrix is positive definite using svd
            u, s, v = np.linalg.svd(q_mat_d)
            psd = np.allclose(np.dot(v.T * s, v), q_mat_d)
            stochasticity_ok[n] = psd
    if bounds_dict is None and not stochastic_run:
        return positive & physical
    elif bounds_dict is not None and not stochastic_run:
        return positive & physical & inside_bounds
    elif bounds_dict is None and stochastic_run:
        return positive & physical & stochasticity_ok
    else:
        return positive & physical & inside_bounds & stochasticity_ok

def get_constraint_ranges():
    return {'ECS': (1.0,7.0),
            'TCR': (0.5,3.5),
            'Tinc': (0.15,2.32),
            'ERFari': (-1.0,0.4),
            'ERFaci': (-2.3,0.3),
            'ERFaer': (-3.0,0.0),
            'CO2conc2022': (410.0,424.0),
            'OHC': (0, 900),
            'Tvar': (0.0,0.150)}

def get_param_ranges():
    '''
    return {'clim_gamma': (0.8,28.0), 'clim_c1': (2.0,9.6), 'clim_c2':(2.0,60.0), 'clim_c3':(2.5,260.0),
            'clim_kappa1': (0.301,3.0), 'clim_kappa2': (0.01,7.5), 'clim_kappa3': (0.01,2.5), 'clim_epsilon': (0.02,2.6),
            'clim_sigma_eta': (0.01,2.6),'clim_sigma_xi':(0.01,1.0),'clim_F_4xCO2':(5.0,11.5),'cc_r0':(20.0,50.0),
            'cc_rU': (-0.025,0.025),'cc_rT': (-2.7,7.5),'cc_rA':(-0.01,0.012),'ari_BC':(0.0,0.05),
            'ari_CH4': (-4.54e-6,0.0),'ari_N2O': (-7.27e-5,0.0), 'ari_NH3': (-8e-4,-4.5e-4),
            'ari_NOx': (-0.000135,0.0),'ari_OC': (-0.01,0.002), 'ari_Sulfur': (-0.008,0.002),
            'ari_VOC': (-4e-5,1e-5),'ari_Ee_stratospheric_Cl': (-1.2e-5,-0.4e-5),
            'aci_shape_SO2': (0.001,0.07),'aci_shape_BC': (0.0,0.07),'aci_shape_OC': (0.0,0.07),
            'aci_beta':(-3.0,0.0),'o3_CH4': (1e-4,4e-4),'o3_N2O': (0.0,2e-3),'o3_Ee_stratospheric_Cl': (-3e-4,1.5e-4),
            'o3_CO': (-2e-4,3e-4),'o3_VOC': (-4e-4,9e-4),'o3_NOx': (-0.5e-3,3.5e-3), 'fscale_CH4': (0.5,1.5),
            'fscale_N2O': (0.7,1.3), 'fscale_minorGHG': (0.6,1.4), 'fscale_stratospheric_H2O_vapor': (-0.5,2.5),
            'fscale_light_abs_particles': (-0.6,2.6),'fscale_Land use': (0.0,2.0),
            'fscale_Volcanic': (0.5,1.5), 'fscale_solar_amplitude': (0.0,2.0),'fscale_solar_trend': (-0.12,0.12),
            'fscale_CO2': (0.75,1.25), 'cc_CO2_conc_1750': (270,286),'seed': (0,int(6e8))}
    '''
    return {'clim_gamma': (0.5,27.0), 'clim_c1': (2.0,7.0), 'clim_c2':(2.0,50.0), 'clim_c3':(2.5,260.0),
            'clim_kappa1': (0.3,2.9), 'clim_kappa2': (0.01,7.5), 'clim_kappa3': (0.01,3.5), 'clim_epsilon': (0.01,2.4),
            'clim_sigma_eta': (0.01,2.8),'clim_sigma_xi':(0.01,1.25),'clim_F_4xCO2':(5.0,11.0),
            'cc_r0':(15.0,55.0),'cc_rU': (-0.008,0.017),'cc_rT': (-2.7,7.7),'cc_rA':(-0.005,0.008),
            'ari_BC':(-0.02,0.08),'ari_OC': (-0.012,0.003), 'ari_Sulfur': (-0.008,0.002), 'ari_NOx': (-0.00015,-0.00002),
            'ari_VOC': (-6e-5,3e-5),'ari_NH3': (-0.00080,-0.00045),'ari_CH4': (-6e-6,1e-6),'ari_N2O': (-8e-5,1e-5), 
            'ari_Ee_stratospheric_Cl': (-1.2e-5,-0.5e-5),'log(-aci_beta)':(-1.5,4.0),'log(aci_shape_SO2)': (-11.0,2.0),
            'log(aci_shape_BC)': (-11.0,1.0),'log(aci_shape_OC)': (-120.0,25.0),'o3_CH4': (0.00010,0.00035),'o3_N2O': (2e-4,22e-4),
            'o3_Ee_stratospheric_Cl': (-3.0e-4,1.5e-4),'o3_CO': (-2.5e-4,2.5e-4),'o3_VOC': (-5e-4,10e-4),'o3_NOx': (-1e-3,3.5e-3),
            'fscale_CH4': (0.5,1.5),'fscale_N2O': (0.7,1.3), 'fscale_minorGHG': (0.6,1.4), 'fscale_stratospheric_H2O_vapor': (-1.0,3.0),
            'fscale_Land use': (0.0,2.0),'fscale_Volcanic': (0.4,1.6), 'fscale_solar_amplitude': (0.0,2.0), 'fscale_solar_trend': (-0.14,0.14),
            'fscale_light_abs_particles': (-1.0,3.0),'fscale_CO2': (0.7,1.3),'cc_CO2_conc_1750': (273,285),'seed': (0,int(6e8))}

def get_constraint_targets():
    def opt(x, q05_desired, q50_desired, q95_desired):
        "x is (a, loc, scale) in that order."
        q05, q50, q95 = skewnorm.ppf((0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2])
        return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)
    # Optimize with scipy
    ecs_params = root(opt, x0=np.ones(3), args=(2, 3, 5)).x
    gsat_params = root(opt, x0=np.ones(3), args=(0.87, 1.03, 1.13)).x
    
    NINETY_TO_SIGMA = np.sqrt(chi2.ppf(0.9,1))
    # Target functions
    out = {}
    out["ECS"] = lambda x: skewnorm.pdf(x,ecs_params[0], loc=ecs_params[1], scale=ecs_params[2])
    out["TCR"] = lambda x: norm.pdf(x, loc=1.8, scale=0.6 / NINETY_TO_SIGMA)
    out["Tinc"] = lambda x: skewnorm.pdf(x, gsat_params[0],loc=gsat_params[1],scale=gsat_params[2])
    out["ERFari"] = lambda x: norm.pdf(x, loc=-0.3, scale=0.3 / NINETY_TO_SIGMA)
    out["ERFaci"] = lambda x: norm.pdf(x, loc=-1.0, scale=0.7 / NINETY_TO_SIGMA)
    out["ERFaer"] = lambda x: norm.pdf(x, loc=-1.3, scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_SIGMA)
    out["CO2conc2022"] = lambda x: norm.pdf(x, loc=417.0, scale=0.5)
    T = read_temperature_obs(gmst_obs=False,return_unc=False,realizations=True)
    variation = temperature_variation(T)
    variation_mean = variation.mean(dim='realization').data
    variation_std = variation.std(dim='realization',ddof=1).data
    out["OHC"] = lambda x: norm.pdf(x, loc=465.3, scale=108.5)
    out["Tvar"] = lambda x: norm.pdf(x,loc=variation_mean,scale=variation_std)
    return out

def get_log_constraint_target():
    def opt(theta, q05_desired, q50_desired, q95_desired):
        "theta is (a, loc, scale) in that order."
        q05, q50, q95 = skewnorm.ppf((0.05, 0.50, 0.95), theta[0], loc=theta[1], scale=theta[2])
        return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)
    # Optimize with scipy
    ecs_params = root(opt, x0=np.ones(3), args=(2, 3, 5)).x
    gsat_params = root(opt, x0=np.ones(3), args=(0.87, 1.03, 1.13)).x
    # Target for temperature variation from measured data
    T = read_temperature_obs(gmst_obs=False,return_unc=False,realizations=True)
    variation = temperature_variation(T)
    variation_mean = variation.mean(dim='realization').data
    variation_std = variation.std(dim='realization',ddof=1).data
    
    NINETY_TO_SIGMA = np.sqrt(chi2.ppf(0.9,1))
    # Log-target
    return lambda x: skewnorm.logpdf(x[:,0],ecs_params[0],loc=ecs_params[1],scale=ecs_params[2]) +\
                     norm.logpdf(x[:,1],loc=1.8,scale=0.6/NINETY_TO_SIGMA) +\
                     skewnorm.logpdf(x[:,2],gsat_params[0],loc=gsat_params[1],scale=gsat_params[2]) +\
                     norm.logpdf(x[:,3],loc=-0.3,scale=0.3/NINETY_TO_SIGMA) +\
                     norm.logpdf(x[:,4],loc=-1.0,scale=0.7/NINETY_TO_SIGMA) +\
                     norm.logpdf(x[:,5],loc=-1.3,scale=np.sqrt(0.7**2+0.3**2)/NINETY_TO_SIGMA) +\
                     norm.logpdf(x[:,6],loc=417.0,scale=0.5) +\
                     norm.logpdf(x[:,7],loc=465.3,scale=108.5) +\
                     norm.logpdf(x[:,8],loc=variation_mean,scale=variation_std)
    
def resample_constraint_posteriors(N=10**5):
    # The location of 95% quantile
    NINETY_TO_SIGMA = np.sqrt(chi2.ppf(0.9,1))
    valid_temp = np.loadtxt(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/runids_rmse_pass.csv").astype(int)

    input_ensemble_size = len(valid_temp)

    assert input_ensemble_size > output_ensemble_size

    temp_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy")
    ohc_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ocean_heat_content_2020_minus_1971.npy")
    fari_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ari_2005-2014_mean.npy")
    faci_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_aci_2005-2014_mean.npy")
    co2_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/concentration_co2_2022.npy")
    ecs_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy")
    tcr_in = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy")
    faer_in = fari_in + faci_in

    def opt(x, q05_desired, q50_desired, q95_desired):
        "x is (a, loc, scale) in that order."
        q05, q50, q95 = skewnorm.ppf((0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2])
        return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)
    
    ecs_params = root(opt, x0=np.ones(3), args=(2, 3, 5)).x
    gsat_params = root(opt, x0=np.ones(3), args=(0.87, 1.03, 1.13)).x
    
    #samples = read_emissions(start,end,scenario='ssp245',nconfigs=1)
    samples = {}
    samples["ECS"] = skewnorm.rvs(ecs_params[0], loc=ecs_params[1], scale=ecs_params[2], size=N, random_state=91603)
    samples["TCR"] = norm.rvs(loc=1.8, scale=0.6 / NINETY_TO_SIGMA, size=N, random_state=18196)
    samples["Tinc"] = skewnorm.rvs(gsat_params[0],loc=gsat_params[1],scale=gsat_params[2],size=10**5,random_state=19387)
    samples["ERFari"] = norm.rvs(loc=-0.3, scale=0.3 / NINETY_TO_SIGMA, size=10**5, random_state=70173)
    samples["ERFaci"] = norm.rvs(loc=-1.0, scale=0.7 / NINETY_TO_SIGMA, size=N, random_state=91123)
    samples["ERFaer"] = norm.rvs(loc=-1.3, scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_SIGMA, size=10**5, random_state=3916153)
    samples["CO2conc2022"] = norm.rvs(loc=417.0, scale=0.5, size=10**5, random_state=81693) # IGCC paper: 417.1 +/- 0.4, IGCC dataset: 416.9, C. Smith: 417.0 +/- 0.5
    samples["OHC"] = norm.rvs(loc=465.3, scale=108.5, size=N, random_state=43178)

    ar_distributions = {}
    for constraint in ['ECS','TCR','Tinc','ERFari','ERFaci','ERFaer','CO2conc2022','OHC']:
        ar_distributions[constraint] = {}
        ar_distributions[constraint]["bins"] = np.histogram(samples[constraint], bins=100, density=True)[1]
        ar_distributions[constraint]["values"] = samples[constraint]

    weights_20yr = np.concatenate(([0.5],np.ones(19),[0.5])) / 20
    weights_50yr = np.concatenate(([0.5],np.ones(49),[0.5])) / 50

    accepted = DataFrame(
        {
            "ECS": ecs_in[valid_temp],
            "TCR": tcr_in[valid_temp],
            "Tinc": np.average(temp_in[153:174, valid_temp], weights=weights_20yr, axis=0) -\
                    np.average(temp_in[:51, valid_temp], weights=weights_50yr, axis=0),
            "ERFari": fari_in[valid_temp],
            "ERFaci": faci_in[valid_temp],
            "ERFaer": faer_in[valid_temp],
            "CO2conc2022": co2_in[valid_temp],
            "OHC": ohc_in[valid_temp] / 1e21
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
            DataFrame(np.array(gofs), columns=unique_codes),
            DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
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
                    assessed_ranges_bin_counts[histogram_idx] / existing_weighted_bin_counts[histogram_idx]
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

def setup_fair(scenarios, nconfigs, start=1750, end=2024, dtype=np.float64):
    '''
    Parameters
    ----------
    scenarios : list of str
        FaIR scenarios
    nconfigs : int
        Number of configs to be run
    
    Returns
    -------
    f : FaIR run object
    '''
    df_methane = read_csv(f'{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/CH4_lifetime.csv',
                          index_col=0)
    df_landuse = read_csv(f'{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/landuse_scale_factor.csv',
                          index_col=0)
    df_lapsi = read_csv(f'{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/lapsi_scale_factor.csv',
                        index_col=0)
    emissions = read_emissions(start,end,scenarios,nconfigs=nconfigs, dtype=dtype)
    # Initiate FAIR object
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(start, end, 1)
    f.define_scenarios(scenarios)
    f.define_configs(np.arange(nconfigs,dtype=int))
    species, properties = read_properties()
    species.remove("Halon-1202")
    species.remove("NOx aviation")
    species.remove("Contrails")
    f.define_species(species, properties)
    # Allocate
    f.allocate(dtype=dtype)
    # Emissions
    f.emissions = emissions
    # species level
    f.fill_species_configs()
    # methane lifetime baseline and sensitivity
    fill(f.species_configs["unperturbed_lifetime"],df_methane.loc["historical_best", "base"],specie="CH4")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "CH4"],specie="CH4")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "N2O"],specie="N2O")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "VOC"],specie="VOC")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "NOx"],specie="NOx")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "HC"],specie="Equivalent effective stratospheric chlorine")
    fill(f.species_configs["lifetime_temperature_sensitivity"],df_methane.loc["historical_best", "temp"])
    # correct land use and LAPSI scale factor terms
    fill(f.species_configs["land_use_cumulative_emissions_to_forcing"],df_landuse.loc["historical_best", "CO2_AFOLU"],specie="CO2 AFOLU")
    fill(f.species_configs["lapsi_radiative_efficiency"],df_lapsi.loc["historical_best", "BC"],specie="BC")
    # emissions adjustments for N2O and CH4
    fill(f.species_configs["baseline_emissions"], 38.246272, specie="CH4")
    fill(f.species_configs["baseline_emissions"], 0.92661989, specie="N2O")
    fill(f.species_configs["baseline_emissions"], 19.41683292, specie="NOx")
    fill(f.species_configs["baseline_emissions"], 2.293964929, specie="Sulfur")
    fill(f.species_configs["baseline_emissions"], 348.4549732, specie="CO")
    fill(f.species_configs["baseline_emissions"], 60.62284009, specie="VOC")
    fill(f.species_configs["baseline_emissions"], 2.096765609, specie="BC")
    fill(f.species_configs["baseline_emissions"], 15.44571911, specie="OC")
    fill(f.species_configs["baseline_emissions"], 6.656462698, specie="NH3")
    fill(f.species_configs["baseline_emissions"], 0.92661989, specie="N2O")
    fill(f.species_configs["baseline_emissions"], 0.02129917, specie="CCl4")
    fill(f.species_configs["baseline_emissions"], 202.7251231, specie="CHCl3")
    fill(f.species_configs["baseline_emissions"], 211.0095537, specie="CH2Cl2")
    fill(f.species_configs["baseline_emissions"], 4544.519056, specie="CH3Cl")
    fill(f.species_configs["baseline_emissions"], 111.4920237, specie="CH3Br")
    fill(f.species_configs["baseline_emissions"], 0.008146006, specie="Halon-1211")
    fill(f.species_configs["baseline_emissions"], 0.000010554155, specie="SO2F2")
    fill(f.species_configs["baseline_emissions"], 0, specie="CF4")
    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    #f.run(progress=False)
    return f

def run_configs(f_init,df_configs,solar_forcing,volcanic_forcing,
                start=1750,end=2024,stochastic_run=True,use_seed=True):
    if start < 1750:
        raise ValueError('Start year earlier than 1750 not supported')
    
    trend_shape = np.ones(end-start+1)
    # Trend increase from the year 1750 to the year 2021
    trend_increase = np.linspace(0, 1, 271)
    if end <= 2020:
        trend_shape[:(min(end,2021)-start)] = trend_increase[len(trend_increase)-(min(end,2021)-start):]
    # solar and volcanic forcing
    fill(f_init.forcing,volcanic_forcing[:, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),specie="Volcanic")
    fill(f_init.forcing,solar_forcing[:, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze() + \
         trend_shape[:, None, None] * df_configs["fscale_solar_trend"].values.squeeze(),specie="Solar")
    # climate response
    fill(f_init.climate_configs["ocean_heat_capacity"], df_configs.loc[:, "clim_c1":"clim_c3"].values)
    fill(f_init.climate_configs["ocean_heat_transfer"], df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values)
    fill(f_init.climate_configs["deep_ocean_efficacy"], df_configs["clim_epsilon"].values.squeeze())
    fill(f_init.climate_configs["gamma_autocorrelation"],df_configs["clim_gamma"].values.squeeze())
    fill(f_init.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
    fill(f_init.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
    fill(f_init.climate_configs["seed"], df_configs["seed"])
    fill(f_init.climate_configs["stochastic_run"], stochastic_run)
    fill(f_init.climate_configs["use_seed"], use_seed)
    fill(f_init.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])
    # carbon cycle
    fill(f_init.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(),specie="CO2")
    fill(f_init.species_configs["iirf_airborne"],df_configs["cc_rA"].values.squeeze(),specie="CO2")
    fill(f_init.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(),specie="CO2")
    fill(f_init.species_configs["iirf_temperature"],df_configs["cc_rT"].values.squeeze(),specie="CO2")
    # aerosol indirect
    fill(f_init.species_configs["aci_scale"],df_configs["aci_beta"].values.squeeze())
    fill(f_init.species_configs["aci_shape"],df_configs["aci_shape_SO2"].values.squeeze(),specie="Sulfur")
    fill(f_init.species_configs["aci_shape"],df_configs["aci_shape_BC"].values.squeeze(),specie="BC")
    fill(f_init.species_configs["aci_shape"],df_configs["aci_shape_OC"].values.squeeze(),specie="OC")
    # aerosol direct
    for specie in ["BC","CH4","N2O","NH3","NOx","OC","Sulfur","VOC","Equivalent effective stratospheric chlorine"]:
        if specie == "Equivalent effective stratospheric chlorine":
            config_specie = "Ee_stratospheric_Cl"
        else:
            config_specie = specie
        fill(f_init.species_configs["erfari_radiative_efficiency"],df_configs[f"ari_{config_specie}"],specie=specie)
    # forcing scaling
    for specie in ["CO2","CH4","N2O","Stratospheric water vapour","Light absorbing particles on snow and ice","Land use"]:
        if specie == "Stratospheric water vapour":
            config_specie = "stratospheric_H2O_vapor"
        elif specie == "Light absorbing particles on snow and ice":
            config_specie = "light_abs_particles"
        else:
            config_specie = specie
        fill(f_init.species_configs["forcing_scale"],df_configs[f"fscale_{config_specie}"].values.squeeze(),specie=specie)
    # different species
    for specie in ["CFC-11","CFC-12","CFC-113","CFC-114","CFC-115",
                   "HCFC-22","HCFC-141b","HCFC-142b","CCl4",
                   "CHCl3","CH2Cl2","CH3Cl","CH3CCl3","CH3Br",
                   "Halon-1211","Halon-1301","Halon-2402",
                   "CF4","C2F6","C3F8","c-C4F8","C4F10","C5F12","C6F14","C7F16","C8F18",
                   "NF3","SF6","SO2F2","HFC-125","HFC-134a","HFC-143a","HFC-152a",
                   "HFC-227ea","HFC-23","HFC-236fa","HFC-245fa","HFC-32","HFC-365mfc"]:
        fill(f_init.species_configs["forcing_scale"],df_configs["fscale_minorGHG"].values.squeeze(),specie=specie)
    # ozone
    for specie in ["CH4","N2O","Equivalent effective stratospheric chlorine","CO","VOC","NOx",]:
        if specie == "Equivalent effective stratospheric chlorine":
            config_specie = "Ee_stratospheric_Cl"
        else:
            config_specie = specie
        fill(f_init.species_configs["ozone_radiative_efficiency"],df_configs[f"o3_{config_specie}"],specie=specie)
    # tune down volcanic efficacy
    fill(f_init.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")
    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(f_init.species_configs["baseline_concentration"],df_configs["cc_CO2_conc_1750"].values.squeeze(),specie="CO2")
    f_init.run(progress=False)
    return f_init

def run_fair(df_configs, scenarios, solar_forcing, volcanic_forcing, emissions, df_methane, df_landuse, df_lapsi,
             start=1750, end=2020, stochastic_run=True):
    '''
    Parameters
    ----------
    df_configs : Pandas DataFrame
        Parameter dataframe. Each row correspond to one set of parameters.
    scenario : str
        FaIR scenario
    solar_forcing : Numpy array
        Solar forcing data
    volcanic_forcing : Numpy array
        Volcanic forcing data
    emissions : xarray DataArray
        Emissions for FaIR experiments
    
    Returns
    -------
    f : FaIR run object
    '''
    
    if start < 1750:
        raise ValueError('Start year earlier than 1750 not supported')
    
    trend_shape = np.ones(end-start+1)
    # Trend increase from the year 1750 to the year 2021
    trend_increase = np.linspace(0, 1, 271)
    if start <= 2020:
        trend_shape[:(min(end,2021)-start)] = trend_increase[len(trend_increase)-(min(end,2021)-start):]
        
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(start, end, 1)
    f.define_scenarios(scenarios)
    f.define_configs(df_configs.index)
    species, properties = read_properties()
    species.remove("Halon-1202")
    species.remove("NOx aviation")
    species.remove("Contrails")
    f.define_species(species, properties)
    
    f.allocate()
    
    # Emissions
    f.emissions = emissions
    
    # solar and volcanic forcing
    fill(f.forcing,volcanic_forcing[:, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),specie="Volcanic")
    fill(f.forcing,solar_forcing[:, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze() + \
         trend_shape[:, None, None] * df_configs["fscale_solar_trend"].values.squeeze(),specie="Solar")
    
    # climate response
    fill(f.climate_configs["ocean_heat_capacity"],df_configs.loc[:, "clim_c1":"clim_c3"].values)
    fill(f.climate_configs["ocean_heat_transfer"],df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values)
    fill(f.climate_configs["deep_ocean_efficacy"],df_configs["clim_epsilon"].values.squeeze())
    fill(f.climate_configs["gamma_autocorrelation"],df_configs["clim_gamma"].values.squeeze())
    fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
    fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
    fill(f.climate_configs["seed"], df_configs["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])

    # species level
    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_airborne"],df_configs["cc_rA"].values.squeeze(),specie="CO2")
    fill(f.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_temperature"],df_configs["cc_rT"].values.squeeze(),specie="CO2")

    # aerosol indirect
    fill(f.species_configs["aci_scale"], df_configs["aci_beta"].values.squeeze())
    fill(f.species_configs["aci_shape"],df_configs["aci_shape_so2"].values.squeeze(),specie="Sulfur")
    fill(f.species_configs["aci_shape"],df_configs["aci_shape_bc"].values.squeeze(),specie="BC")
    fill(f.species_configs["aci_shape"],df_configs["aci_shape_oc"].values.squeeze(),specie="OC")

    # methane lifetime baseline and sensitivity
    fill(f.species_configs["unperturbed_lifetime"],df_methane.loc["historical_best", "base"],specie="CH4")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "CH4"],specie="CH4")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "N2O"],specie="N2O")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "VOC"],specie="VOC")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "NOx"],specie="NOx")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"],df_methane.loc["historical_best", "HC"],specie="Equivalent effective stratospheric chlorine")
    fill(f.species_configs["lifetime_temperature_sensitivity"],df_methane.loc["historical_best", "temp"])

    # correct land use and LAPSI scale factor terms
    fill(f.species_configs["land_use_cumulative_emissions_to_forcing"],df_landuse.loc["historical_best", "CO2_AFOLU"],specie="CO2 AFOLU")
    fill(f.species_configs["lapsi_radiative_efficiency"],df_lapsi.loc["historical_best", "BC"],specie="BC")

    # emissions adjustments for N2O and CH4 (we don't want to make these defaults as people
    # might wanna run pulse expts with these gases)
    fill(f.species_configs["baseline_emissions"], 38.246272, specie="CH4")
    fill(f.species_configs["baseline_emissions"], 0.92661989, specie="N2O")
    fill(f.species_configs["baseline_emissions"], 19.41683292, specie="NOx")
    fill(f.species_configs["baseline_emissions"], 2.293964929, specie="Sulfur")
    fill(f.species_configs["baseline_emissions"], 348.4549732, specie="CO")
    fill(f.species_configs["baseline_emissions"], 60.62284009, specie="VOC")
    fill(f.species_configs["baseline_emissions"], 2.096765609, specie="BC")
    fill(f.species_configs["baseline_emissions"], 15.44571911, specie="OC")
    fill(f.species_configs["baseline_emissions"], 6.656462698, specie="NH3")
    fill(f.species_configs["baseline_emissions"], 0.92661989, specie="N2O")
    fill(f.species_configs["baseline_emissions"], 0.02129917, specie="CCl4")
    fill(f.species_configs["baseline_emissions"], 202.7251231, specie="CHCl3")
    fill(f.species_configs["baseline_emissions"], 211.0095537, specie="CH2Cl2")
    fill(f.species_configs["baseline_emissions"], 4544.519056, specie="CH3Cl")
    fill(f.species_configs["baseline_emissions"], 111.4920237, specie="CH3Br")
    fill(f.species_configs["baseline_emissions"], 0.008146006, specie="Halon-1211")
    fill(f.species_configs["baseline_emissions"], 0.000010554155, specie="SO2F2")
    fill(f.species_configs["baseline_emissions"], 0, specie="CF4")

    # aerosol direct
    for specie in ["BC","CH4","N2O","NH3","NOx","OC","Sulfur","VOC","Equivalent effective stratospheric chlorine"]:
        fill(f.species_configs["erfari_radiative_efficiency"],df_configs[f"ari_{specie}"],specie=specie)

    # forcing scaling
    for specie in ["CO2","CH4","N2O","Stratospheric water vapour","Light absorbing particles on snow and ice","Land use"]:
        fill(f.species_configs["forcing_scale"],df_configs[f"fscale_{specie}"].values.squeeze(),specie=specie)

    for specie in ["CFC-11","CFC-12","CFC-113","CFC-114","CFC-115",
                   "HCFC-22","HCFC-141b","HCFC-142b","CCl4",
                   "CHCl3","CH2Cl2","CH3Cl","CH3CCl3","CH3Br",
                   "Halon-1211","Halon-1301","Halon-2402",
                   "CF4","C2F6","C3F8","c-C4F8","C4F10","C5F12","C6F14","C7F16","C8F18",
                   "NF3","SF6","SO2F2","HFC-125","HFC-134a","HFC-143a","HFC-152a",
                   "HFC-227ea","HFC-23","HFC-236fa","HFC-245fa","HFC-32","HFC-365mfc"]:
        fill(f.species_configs["forcing_scale"],df_configs["fscale_minorGHG"].values.squeeze(),specie=specie)

    # ozone
    for specie in ["CH4","N2O","Equivalent effective stratospheric chlorine","CO","VOC","NOx",]:
        fill(f.species_configs["ozone_radiative_efficiency"],df_configs[f"o3_{specie}"],specie=specie)

    # tune down volcanic efficacy
    fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")


    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(f.species_configs["baseline_concentration"],df_configs["cc_co2_concentration_1750"].values.squeeze(),specie="CO2")

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
        f"{cwd}/fair_calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
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

    return tcre,tcr

def compute_data_loss(model,data_mean,data_var,target='wss'):
    # Weighted square-sum
    if target == 'wss':
        wss = ((model-data_mean)**2/data_var).sum(dim='time')
        loss = 0.5 * (wss + (np.log(2*np.pi*data_var)).sum(dim='time'))
    elif target == 'trend_line':
        start, end = 1900, 2020
        x = np.arange(start,end+1,1)
        y = model.sel(timebounds=slice(start,end)).data
        params = np.polyfit(x,y,1)
        loss = 0.5 * (np.square(params[0]-data_mean)/data_var + np.log(2*np.pi*data_var))
    else:
        raise ValueError(f'Unknown target log likelihood {target}')
    '''
    elif method == 'detrend':
        start, end = 1850, 2020
        y = model.sel(timebounds=slice(start,end+1)).data.T
        x = np.arange(0,end-start+1,1)
        detrended = detrend(x,y)
        std = np.std(detrended)
        loss = 0.5 * (np.square(std-mean)/var + np.log(2*np.pi*var))
    '''
    return loss

def compute_constraints(fair,T_variation=True):    
    constraints = ["ECS","TCR","Tinc","ERFari","ERFaci","ERFaer","CO2conc2022","OHC","Tvar"]
    scenarios, configs = fair.scenarios, fair.configs
    out = xr.DataArray(data=np.full((len(constraints),len(scenarios),len(configs)),np.nan),dims=['constraint','scenario','config'],
                       coords=dict(constraint=(['constraint'],constraints),scenario=(['scenario'],scenarios),config=(['config'],configs)))
    # Equilibrium Climate Sensitivity
    out.loc[dict(constraint='ECS')] = fair.ebms.ecs
    # Transient Climate Response
    out.loc[dict(constraint='TCR')] = fair.ebms.tcr
    # Change between the average temperature between years 2003 and 2022 referenced to the average temperature between years 1850 and 1900
    weights_50yr = xr.DataArray(data=np.concatenate(([0.5],np.ones(49),[0.5]))/50,dims=['timebounds'],
                                coords=dict(timebounds=("timebounds",np.arange(1850,1901,dtype=np.int32))))
    weights_19yr = xr.DataArray(data=np.concatenate(([0.5],np.ones(18),[0.5]))/19,dims=['timebounds'],
                                coords=dict(timebounds=("timebounds",np.arange(2003,2023,dtype=np.int32))))
    avgT_1850_1900 = fair.temperature.sel(timebounds=slice(1850,1900),layer=0).weighted(weights=weights_50yr).sum(dim='timebounds')
    avgT_2003_2022 = fair.temperature.sel(timebounds=slice(2003,2022),layer=0).weighted(weights=weights_19yr).sum(dim='timebounds')
    out.loc[dict(constraint='Tinc')] = avgT_2003_2022 - avgT_1850_1900
    # Weights
    weights_10y = xr.DataArray(data=np.concatenate(([0.5],np.ones(8),[0.5]))/9,dims=['timebounds'],
                              coords=dict(timebounds=("timebounds",np.arange(2005,2015,dtype=np.int32))))
    # Average aerosol-radiation interactions between 2005-2014 (relative to 1750)
    out.loc[dict(constraint='ERFari')] = fair.forcing.sel(timebounds=slice(2005,2014),specie='Aerosol-radiation interactions').weighted(weights=weights_10y).sum(dim='timebounds')
    # Average aerosol-cloud interactions between 2005-2014 (relative to 1750)
    out.loc[dict(constraint='ERFaci')] = fair.forcing.sel(timebounds=slice(2005,2014),specie='Aerosol-cloud interactions').weighted(weights=weights_10y).sum(dim='timebounds')
    # Total aerosol interaction
    out.loc[dict(constraint='ERFaer')] = out.sel(constraint='ERFari') + out.sel(constraint='ERFaci')
    # Average CO2 concentration in year 2022
    out.loc[dict(constraint='CO2conc2022')] = 0.5*fair.concentration.sel(timebounds=2022,specie='CO2') + 0.5*fair.concentration.sel(timebounds=2023,specie='CO2')
    # Ocean heat content change between on year 2020 referenced with the value on 1971
    ohc_1971 = 0.5*fair.ocean_heat_content_change.sel(timebounds=1971) + 0.5*fair.ocean_heat_content_change.sel(timebounds=1972)
    ohc_2020 = 0.5*fair.ocean_heat_content_change.sel(timebounds=2020) + 0.5*fair.ocean_heat_content_change.sel(timebounds=2021)
    out.loc[dict(constraint='OHC')] = (ohc_2020 - ohc_1971) / 1e21
    # Variablity (standard deviation) of the detrended temperature (at surface layer)
    if T_variation:
        T = fair.temperature.sel(timebounds=slice(1850,2023),layer=0)
        out.loc[dict(constraint='Tvar')] = temperature_variation(T-avgT_1850_1900)
    return out

def temperature_anomaly(fair,start=1850,end=2023,rolling_window_size=1,layer=0):
    if fair.timebounds.min() > 1850 or fair.timebounds.max() < 1900:
        raise ValueError('Invalid range for timebounds')
    ref_years = np.arange(1850,1901,dtype=np.uint32)
    weights_50yr = xr.DataArray(data=np.concatenate(([0.5],np.ones(49),[0.5]))/50,dims=['timebounds'],coords=dict(timebounds=("timebounds",ref_years)))              
    ref_T = fair.temperature.sel(timebounds=ref_years,layer=layer).drop_vars(['layer']).weighted(weights=weights_50yr).sum(dim='timebounds')
    T = fair.temperature.sel(timebounds=slice(start,end),layer=layer).drop_vars(['layer']) - ref_T
    timebounds = T.timebounds.data
    T = T.rename({'timebounds':'time'})
    new_time_coords = np.convolve(timebounds,np.ones(rolling_window_size)/rolling_window_size,mode='valid')
    new_coords = dict(zip(['time'] + [dim for dim in T.dims if dim != 'time'], 
                          [new_time_coords] + [T[dim].data for dim in T.dims if dim != 'time']))
    T = T.rolling({'time':rolling_window_size},center=True).mean().dropna('time').assign_coords(new_coords)
    return T

def temperature_variation(T,deg=4):
    if 'timebounds' in T.dims:
        T = T.rename({'timebounds':'time'})
    # Assume float-type time coordinates as {year}.{fractional_year}
    timestamps = T.time.data
    coef_ds = T.polyfit('time',deg)
    time_xr = xr.DataArray(data=timestamps,dims=['time'],coords=dict(time=(['time'],timestamps)))
    # Fitted temperature
    fit = xr.dot(time_xr ** coef_ds.degree, coef_ds['polyfit_coefficients'])
    # Detrend the data by subtracting the fitted temperature and save the std as the constraint
    detrended = T - fit
    #fit.sel(scenario='ssp245',config=0).plot();T_da.sel(scenario='ssp245',config=0).plot();
    return detrended.std(dim='time')

def compute_trends(data,xdim,start=1900,end=2020):
    if not isinstance(data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")
    x = np.arange(start,end+1,1)
    y = data.loc[{xdim:slice(start,end)}].data.T
    params = np.polyfit(x,y,1)
    return params[0]

def get_log_prior(included):
    prior_folder = f'{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors'
    # Kernel density estimation for climate response parameters
    df_cr = read_csv(f'{prior_folder}/climate_response_ebm3.csv')
    for param in df_cr.columns:
        if f'clim_{param}' not in included:
            df_cr.drop(labels=param,axis='columns',inplace=True)
    cr_d = gaussian_kde(df_cr.T) if len(df_cr.columns) != 0 else lambda x: 1.0
    
    # Kernel density estimation for carbon cycle parameters and CO2 level at 1750
    df_cc = read_csv(f"{prior_folder}/carbon_cycle.csv")
    for param in df_cc.columns:
        if f'cc_{param}' not in included:
            df_cc.drop(labels=param,axis='columns',inplace=True)
    cc_d = gaussian_kde(df_cc.T) if len(df_cc.columns) != 0 else lambda x: 1.0
    
    # Aerosol-radiation interaction parameters
    df_ari = read_csv(f"{prior_folder}/aerosol_radiation.csv")
    df_ari = df_ari.rename({'Equivalent effective stratospheric chlorine':'Ee_stratospheric_Cl'},axis='columns')
    for param in df_ari.columns:
        if f'ari_{param}' not in included:
            df_ari.drop(labels=param,axis='columns',inplace=True)
            
    ari_d = gaussian_kde(df_ari.T) if len(df_ari.columns) != 0 else lambda x: 1.0
    
    # Aerosol-cloud interaction parameters
    df_aci = read_csv(f"{prior_folder}/aerosol_cloud.csv")
    df_aci = transform_df(df_aci, 
                          transformation_dict = {'beta': lambda x: np.log(-x),
                                                 'shape_SO2': lambda x: np.log(x),
                                                 'shape_BC': lambda x: np.log(x),
                                                 'shape_OC': lambda x: np.log(x)},
                          name_changes = {'beta': 'log(-aci_beta)',
                                          'shape_SO2': 'log(aci_shape_SO2)',
                                          'shape_BC': 'log(aci_shape_BC)',
                                          'shape_OC': 'log(aci_shape_OC)'})
    
    #df_aci.rename({col: f"shape_{col.split('_')[-1].upper()}" for col in df_aci.columns if 'shape' in col},
    #              axis='columns',inplace=True)
    for param in df_aci.columns:
        if param not in included:
            df_aci.drop(labels=param,axis='columns',inplace=True)
    aci_d = gaussian_kde(df_aci.T) if len(df_aci.columns) != 0 else lambda x: 1.0
    
    # Ozone interaction parameters
    df_o3 = read_csv(f"{prior_folder}/ozone.csv").rename({'Equivalent effective stratospheric chlorine':'Ee_stratospheric_Cl'},axis='columns')
    for param in df_o3.columns:
        if f'o3_{param}' not in included:
            df_o3.drop(labels=param,axis='columns',inplace=True)
    o3_d = gaussian_kde(df_o3.T) if len(df_o3.columns) != 0 else lambda x: 1.0
    
    # Scaling factor parameters
    df_scaling = read_csv(f"{prior_folder}/forcing_scaling.csv").rename({'Stratospheric water vapour':'stratospheric_H2O_vapor',
                                                                         'Light absorbing particles on snow and ice': 'light_abs_particles'},
                                                                         axis='columns')
    for param in df_scaling.columns:
        if f'fscale_{param}' not in included:
            df_scaling.drop(labels=param,axis='columns',inplace=True)
    scaling_d = gaussian_kde(df_scaling.T) if len(df_scaling.columns) != 0 else lambda x: 1.0
    
    # CO2 concentration on year 1750
    df_co2 = read_csv(f"{prior_folder}/co2_concentration_1750.csv").rename({'co2_concentration': 'CO2_conc_1750'},axis='columns')
    for param in df_co2.columns:
        if f'cc_{param}' not in included:
            df_co2.drop(labels=param,axis='columns',inplace=True)
    co2conc_d = gaussian_kde(df_co2.T) if len(df_co2.columns) != 0 else lambda x: 1.0
    
    N_params = [len(df_cr.columns),len(df_cc.columns),len(df_ari.columns),len(df_aci.columns),len(df_o3.columns),len(df_scaling.columns),len(df_co2.columns)]
    #print(df_cr.columns,df_cc.columns,df_ari.columns,df_aci.columns,df_o3.columns,df_scaling.columns,df_co2.columns,sep='\n')
    if sum(N_params) != len(included):
        raise ValueError(f'Number of parameters (={sum(N_params)}) in the prior mismatches the number of the included parameters (={len(included)})')
    
    return lambda x: cr_d.logpdf(x[:,0:sum(N_params[:1])].T) + cc_d.logpdf(x[:,sum(N_params[:1]):sum(N_params[:2])].T) + ari_d.logpdf(x[:,sum(N_params[:2]):sum(N_params[:3])].T) \
                   + aci_d.logpdf(x[:,sum(N_params[:3]):sum(N_params[:4])].T) + o3_d.logpdf(x[:,sum(N_params[:4]):sum(N_params[:5])].T) \
                   + scaling_d.logpdf(x[:,sum(N_params[:5]):sum(N_params[:6])].T) + co2conc_d.logpdf(x[:,sum(N_params[:6]):sum(N_params[:7])].T)
                            
def constraint_priors():
    out = {}
    temp = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy")
    prior_ecs_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ecs.npy")
    out['ECS'] = gaussian_kde(prior_ecs_samples)
    prior_tcr_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/tcr.npy")
    out['TCR'] = gaussian_kde(prior_tcr_samples)
    weights_50y = np.concatenate(([0.5],np.ones(49),[0.5]))/50
    weights_19y = np.concatenate(([0.5],np.ones(18),[0.5]))/19
    out['Tinc'] = gaussian_kde(np.average(temp[153:173,:],axis=0,weights=weights_19y) - np.average(temp[:51,:],axis=0,weights=weights_50y))
    prior_ari_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_ari_2005-2014_mean.npy")
    out['ERFari'] = gaussian_kde(prior_ari_samples)
    prior_aci_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/forcing_aci_2005-2014_mean.npy")
    out['ERFaci'] = gaussian_kde(prior_aci_samples)
    prior_aer_samples = prior_aci_samples + prior_ari_samples
    out['ERFaer'] = gaussian_kde(prior_aer_samples)
    prior_co2_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/concentration_co2_2022.npy")
    out['CO2conc2022'] = gaussian_kde(prior_co2_samples)
    prior_ohc_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/ocean_heat_content_2020_minus_1971.npy") / 1e21
    out['OHC'] = gaussian_kde(prior_ohc_samples)
    prior_variation_samples = np.load(f"{fair_calib_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/T_variation_1850-2022.npy")
    out['Tvar'] = gaussian_kde(prior_variation_samples)
    return out
                   

    
