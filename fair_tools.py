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
from scipy.stats import gamma, norm, uniform, skewnorm, gaussian_kde
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from netCDF4 import Dataset
from fitter import Fitter
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

def load_configs():
    df = pd.read_csv(f"{fair_calibration_dir}/output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
                     "posteriors/calibrated_constrained_parameters.csv",
                     index_col=0)
    return df

def constraint_ranges():
    out = {}
    out['ecs'] = (0,8)
    out['tcr'] = (0,4)
    out['T 1995-2014'] = (0.5,1.3)
    out['ari'] = (-1.0,0.3)
    out['aci'] = (-2.25, 0.25)
    out['aer'] = (-3, 0)
    out['CO2'] = (394, 402)
    out['ohc'] = (0, 800)
    out['T 2081-2100'] = (0.8, 3.2)
    return out

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

def fit_priors(df_configs,exclude=[]):
    # Prior density functions
    distributions = {}
    cols = df_configs.columns
    for col in cols:
        fit = Fitter(df_configs[col].to_numpy(),distributions=["norm","gamma","uniform"])
        fit.fit(progress=False)
        best = fit.get_best(method = 'sumsquare_error')
        for dist in best.keys():
            distributions[col] = {'distribution':dist,'params':tuple(best[dist].values())}
    gamma_params = np.stack([distributions[col]['params'] for col in cols if distributions[col]['distribution'] == 'gamma' and col not in exclude])
    norm_params = np.stack([distributions[col]['params'] for col in cols if distributions[col]['distribution'] == 'norm' and col not in exclude])
    uniform_params = np.array([distributions[col]['params'] for col in cols if distributions[col]['distribution'] == 'uniform' and col not in exclude])
    is_gamma = np.array([distributions[col]['distribution'] == 'gamma' for col in cols if col not in exclude], dtype=bool)
    is_norm = np.array([distributions[col]['distribution'] == 'norm' for col in cols if col not in exclude], dtype=bool)
    is_uniform = np.array([distributions[col]['distribution'] == 'uniform' for col in cols if col not in exclude], dtype=bool)
    is_gamma_indices = np.where(is_gamma)[0]
    is_norm_indices = np.where(is_norm)[0]
    is_uniform_indices = np.where(is_uniform)[0]
    #fun = lambda x: np.prod(uniform.pdf(x[is_uniform],uniform_params[:,0],uniform_params[:,1])) * np.prod(gamma.pdf(x[is_gamma],gamma_params[:,0],gamma_params[:,1],gamma_params[:,2])) * np.prod(norm.pdf(x[is_norm],norm_params[:,0],norm_params[:,1]))
    vec = np.full(len(cols)-len(exclude),np.nan)
    def fun(x):
        np.put(vec,is_gamma_indices,gamma.pdf(x[is_gamma],gamma_params[:,0],gamma_params[:,1],gamma_params[:,2]))
        np.put(vec,is_norm_indices,norm.pdf(x[is_norm],norm_params[:,0],norm_params[:,1]))
        np.put(vec,is_uniform_indices,uniform.pdf(x[is_uniform],uniform_params[:,0],uniform_params[:,1]))
        return vec
    return fun, distributions
    
def get_prior():
    '''
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/4xCO2_cummins_ebm3_cmip6.csv")
    models = df["model"].unique()
    n_models = len(models)
    params = {}
    multi_runs = {
    "GISS-E2-1-G": "r1i1p1f1",
    "GISS-E2-1-H": "r1i1p3f1",
    "MRI-ESM2-0": "r1i1p1f1",
    "EC-Earth3": "r3i1p1f1",
    "FIO-ESM-2-0": "r1i1p1f1",
    "CanESM5": "r1i1p2f1",
    "FGOALS-f3-L": "r1i1p1f1",
    "CNRM-ESM2-1": "r1i1p1f2",
    }
    params["gamma"] = np.full(n_models,np.nan)
    params["c1"] = np.full(n_models,np.nan)
    params["c2"] = np.full(n_models,np.nan)
    params["c3"] = np.full(n_models,np.nan)
    params["kappa1"] = np.full(n_models,np.nan)
    params["kappa2"] = np.full(n_models,np.nan)
    params["kappa3"] = np.full(n_models,np.nan)
    params["epsilon"] = np.full(n_models,np.nan)
    params["sigma_eta"] = np.full(n_models,np.nan)
    params["sigma_xi"] = np.full(n_models,np.nan)
    params[r"F_4xCO2"] = np.full(n_models,np.nan)
    
    for im, model in enumerate(models):
        if model in multi_runs:
            condition = (df["model"] == model) & (df["run"] == multi_runs[model])
        else:
            condition = df["model"] == model
        params[r"gamma"][im] = df.loc[condition, "gamma"].values[0]
        params["c1"][im], params["c2"][im], params["c3"][im] = df.loc[
            condition, "C1":"C3"
        ].values.squeeze()
        (
            params["kappa1"][im],
            params["kappa2"][im],
            params["kappa3"][im],
        ) = df.loc[condition, "kappa1":"kappa3"].values.squeeze()
        params["epsilon"][im] = df.loc[condition, "epsilon"].values[0]
        params["sigma_eta"][im] = df.loc[condition, "sigma_eta"].values[0]
        params["sigma_xi"][im] = df.loc[condition, "sigma_xi"].values[0]
        params[r"F_4xCO2"][im] = df.loc[condition, "F_4xCO2"].values[0]
    params = pd.DataFrame(params)
    # Remove unphysical parameter combinations
    params[params <= 0] = np.nan
    params['gamma'][params['gamma'] <= 0.8] = np.nan
    params['c1'][params['c1'] <= 2] = np.nan
    params['c2'][params['c2'] <= params['c1']] = np.nan
    params['c3'][params['c3'] <= params['c2']] = np.nan
    params['kappa1'][params['kappa1'] <= 0.3] = np.nan
    params = params[params.notna().all(axis=1)]
    
    # Check that covariance matrix is positive semidefinite
    keep = np.full(len(params),True,dtype=bool)
    for i in range(len(params)):
        ebm = EnergyBalanceModel(
            ocean_heat_capacity=params.iloc[i,[1,2,3]],
            ocean_heat_transfer=params.iloc[i,[4,5,6]],
            deep_ocean_efficacy=params.iloc[i,7],
            gamma_autocorrelation=params.iloc[i,0],
            sigma_xi=params.iloc[i,9],
            sigma_eta=params.iloc[i,8],
            forcing_4co2=params.iloc[i,10],
            stochastic_run=True,
        )
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
        if not np.all(eigh(q_mat_d,eigvals_only=True) > 0):
            keep[i] = False
    params = params[keep]
    '''
    # Kernel density estimation for climate response parameters
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/climate_response_ebm3.csv")
    climate_response_d = gaussian_kde(df.T)
    
    # Kernel density estimation for parameters r0, rU, rT and rA
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/carbon_cycle.csv")
    carbon_cycle_d = gaussian_kde(df.T)
    
    # Aerosol-radiation interaction parameters
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_radiation.csv")
    # Ignore carbon monoxide
    df.drop(labels='CO',axis='columns',inplace=True)
    min_values, max_values = df.min(axis=0).to_numpy(), df.max(axis=0).to_numpy()
    lenghts = max_values - min_values
    ari_d = lambda x: np.prod(uniform.pdf(x,min_values-0.05*lenghts,1.1*lenghts))
    
    # Aerosol-clound interaction parameters
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/aerosol_cloud.csv")
    aci_d = gaussian_kde(df.T)
    
    # Ozone interaction parameters
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv")
    ozone_d = gaussian_kde(df.T)
    
    # Kernel-density estimation for scaling factor parameters
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/forcing_scaling.csv")
    scaling_d = gaussian_kde(df.T)
    
    df = pd.read_csv(f"{cwd}/fair-calibrate/output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/co2_concentration_1750.csv")
    co2conc_d = gaussian_kde(df.T)
    
    prior = lambda x: float(climate_response_d(x[:11]) * carbon_cycle_d(x[11:15]) * ari_d(x[15:24]) * aci_d(x[24:28]) * ozone_d(x[28:34]) * scaling_d(x[34:45]) * co2conc_d(x[45])) if len(x) == 46 else np.nan
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

def constraint_posteriors():
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
    samples["ecs"] = skewnorm.rvs(8.82185594, loc=1.95059779, scale=1.55584604, size=10**5, random_state=91603)
    samples["tcr"] = norm.rvs(loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196)
    samples["ohc"] = norm.rvs(loc=396 / 0.91, scale=67 / 0.91, size=10**5, random_state=43178)
    samples["T 1995-2014"] = skewnorm.rvs(-1.65506091, loc=0.92708099, scale=0.12096636, size=10**5, random_state=19387)
    samples["ari"] = norm.rvs(loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173)
    samples["aci"] = norm.rvs(loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123)
    samples["aer"] = norm.rvs(loc=-1.3,scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_ONESIGMA,size=10**5,random_state=3916153)
    samples["CO2"] = norm.rvs(loc=397.5469792683919, scale=0.36, size=10**5, random_state=81693)
    samples["T 2081-2100"] = skewnorm.rvs(2.20496701, loc=1.4124379, scale=0.60080822, size=10**5, random_state=801693589)

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

    draws = []
    drawn_samples = accepted.sample(
        n=output_ensemble_size, replace=False, weights=weights, random_state=10099
    )
    draws.append((drawn_samples))
    
    return {constraint: gaussian_kde(draws[0][constraint]) for constraint in constraints}
    
def load_MC_samples(ds,N=None,tail=True):
    names = ds['param'][:]
    warmup = ds['warmup'][:]
    chain = ds['chain'][:,:]
    pos_configs = pd.DataFrame(columns=names)
    pos_samples = len(chain) - warmup
    
    if N is None:
        N = pos_samples
    elif N > pos_samples:
        msg = f'Number of tail samples {N} larger than the number of total posterior samples {pos_samples}'
        raise ValueError(msg)
    if tail:
        pos_configs[names] = chain[-N:,:]
    else:
        pos_configs[names] = chain[warmup:(warmup+N),:]
    #pos_configs['sigma_eta'] = np.zeros(N)
    #pos_configs['sigma_xi'] = np.zeros(N)
    pos_configs['seed'] = ds['seeds'][warmup:(warmup+N)]
    return pos_configs

def load_optimal_config(folder,scenario):
    ds = Dataset(f'{cwd}/{folder}/{scenario}/sampling.nc','r')
    names = ds['param'][:]
    MAP = ds['MAP'][:]
    ds.close()
    opt_config = load_configs().median(axis=0).to_frame().transpose()
    opt_config['sigma_eta'], opt_config['sigma_xi'] = 0, 0
    opt_config[names] = MAP
    return opt_config

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
    #print(f.concentration.loc[dict(timebounds=1750,specie='Aerosol-radiation interactions',scenario='ssp245')])
    #print(f.concentration.loc[dict(specie='CO2')])
    #raise ValueError
    
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

