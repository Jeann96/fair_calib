#!/usr/bin/env python
# coding: utf-8

"""Sample aerosol indirect."""

# **Note also** the uniform prior from -2 to 0. A lot of the sublteties here might also
# want to go into the paper.


import glob
import os
from dotenv import load_dotenv

import matplotlib.pyplot as pl
#from plotting_tools import plot_chains
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import pooch
from scipy.stats import gaussian_kde, invgamma, norm
from scipy.optimize import curve_fit,least_squares
from scipy.interpolate import UnivariateSpline
#from sklearn.metrics import r2_score
import xarray as xr
#import warnings
#warnings.simplefilter('error')

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
prior_samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
#plots = False
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

files = glob.glob("../../../../../data/smith2023aerosol/*.csv")

ari = {}
aci = {}
models = []
models_runs = {}
years = {}
for file in files:
    model = os.path.split(file)[1].split("_")[0]
    run = os.path.split(file)[1].split("_")[1]
    models.append(model)
    if run not in models_runs:
        models_runs[model] = []
    models_runs[model].append(run)

models = list(models_runs.keys())

for model in models:
    nruns = 0
    for run in models_runs[model]:
        file = f"../../../../../data/smith2023aerosol/{model}_{run}_aerosol_forcing.csv"
        df = pd.read_csv(file, index_col=0)
        if nruns == 0:
            ari_temp = df["ERFari"].values.squeeze()
            aci_temp = df["ERFaci"].values.squeeze()
        else:
            ari_temp = ari_temp + df["ERFari"].values.squeeze()
            aci_temp = aci_temp + df["ERFaci"].values.squeeze()
        years[model] = df.index + 0.5
        nruns = nruns + 1
    ari[model] = ari_temp / nruns
    aci[model] = aci_temp / nruns


# Calibrate on RCMIP
folders = os.getcwd().split("/")

rcmip_emissions_folder = "/" + os.path.join(*folders[:(folders.index("fair-calibrate")+1)],"data","emissions")
rcmip_exists = any(file.endswith(".csv") and "rcmip-emissions-annual-means" in file for file in os.listdir(rcmip_emissions_folder))

if not rcmip_exists:
    rcmip_emissions_path = pooch.retrieve(
        url="https://zenodo.org/records/4589756/files/"
        "rcmip-emissions-annual-means-v5-1-0.csv",
        known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
        progressbar=progress,
        path=rcmip_emissions_folder,
    )
else:
    rcmip_emissions_filename = next(file for file in os.listdir(rcmip_emissions_folder) if file.endswith(".csv") and "rcmip-emissions-annual-means" in file)
    rcmip_emissions_path = os.path.join(rcmip_emissions_folder,rcmip_emissions_filename)
emis_df = pd.read_csv(rcmip_emissions_path)

bc = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|BC"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)

oc = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|OC"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)

so2 = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|Sulfur"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)

def aci_log(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    aci_1850 = beta * np.log(1 + x[0][0] * n0 + x[1][0] * n1 + x[2][0] * n2)
    return aci - aci_1850

def aci_log_nocorrect(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    return aci

# Same as aci_log but can run parallel samples and handles invalid inputs
# without giving warnings
def model_fun(theta,*args,correction=True):
    so2_data, bc_data, oc_data = args
    log_arg = 1.0 + so2_data[:,np.newaxis] * theta[1] + bc_data[:,np.newaxis] * theta[2] + oc_data[:,np.newaxis] * theta[3]
    valid = np.all(log_arg > 0.0, axis=0) & (theta[0] < 0.0)
    aci_timeseries = np.full(log_arg.shape, np.nan)
    aci_timeseries[:,valid] = theta[0][valid] * np.log(log_arg[:,valid])
    if correction:
        aci_1850 = np.full((len(aci_timeseries),theta.shape[-1]),np.nan)
        aci_1850[:,valid] = theta[0][valid] * np.log(log_arg[0,valid])
        return aci_timeseries - aci_1850
    else:
        return aci_timeseries

# For parallel sampling
def parallel_outer_product(X):
    shape = (len(X),1) if X.ndim == 1 else X.shape
    temp = np.zeros((X.size,shape[-1]))
    indices = (np.arange(X.size,dtype=int),np.repeat(np.arange(shape[-1],dtype=int),shape[0]))
    temp[indices] = X.flatten(order='F')
    return np.einsum('ij,kj->ik',temp,temp)

def transform(x):
    return np.vstack((np.log(-x[0]),np.log(x[1]),np.log(x[2]),np.log(x[3])))

# = -ln(1/|det(J)|) = ln(|det(J)|) where J is the Jacobian matrix of the transformation
'''
def logabsjacdet(y):
    return y.sum(axis=0)
'''

def inv_transform(y):
    return np.vstack((-np.exp(y[0]),np.exp(y[1]),np.exp(y[2]),np.exp(y[3])))

# Log prior
def log_prior(x):
    return norm.logpdf(x[0],loc=0.0,scale=2.5) + \
           norm.logpdf(x[1],loc=-5.0,scale=6.0) +\
           norm.logpdf(x[2],loc=-5.0,scale=2.0) +\
           norm.logpdf(x[3],loc=-50.0,scale=30.0)

'''
# Log prior (of the transformed variable)
def log_transformed_prior(x):
    return lognorm.logpdf(-x[0],2.5,scale=1.0) + \
           lognorm.logpdf(x[1],6.0,scale=np.exp(-5.0)) +\
           lognorm.logpdf(x[2],2.0,scale=np.exp(-5.0)) +\
           lognorm.logpdf(x[3],30.0,scale=np.exp(-50.0))
'''

def AM_sampling(fun,ydata,samples,warmup=None,x0=None,C0=None,var=None,sample_var=False,log_prior=None,
                bounds=None,names=None,scales=None,args=(),progress=False,parallel_chains=1,thinning=1):
    '''
    Adaptive Metropolis Monte-Carlo sampling algorithm.
    '''
    if warmup is None:
        warmup = samples // 10
    N = samples * parallel_chains // thinning
    if bounds is not None:
        bounds = np.array(bounds)
    if x0 is None:
        result = least_squares(lambda x: ydata - fun(x,*args), x0, method='lm', x_scale=scales)
        x0 = result.x
        if C0 is None:
            J = result.jac
            C0 = np.linalg.inv(J.T @ J)
    elif C0 is None and x0 is not None:
        x0 = np.array(x0)
        zeros = np.where(x0 == 0.0)[0]
        if len(zeros) == 0:
            C0 = np.diag(np.square(0.2*np.abs(x0)))
        elif bounds is not None:
            C0 = np.zeros((len(x0),len(x0)))
            non_zeros = np.nonzero(x0)[0]
            C0[non_zeros,non_zeros] = np.square(0.2*np.abs(x0[non_zeros]))
            C0[zeros,zeros] = (0.5 * (bounds[zeros,1] - bounds[zeros,0]) / 3.0) ** 2
        elif scales is not None:
            C0 = np.zeros((len(x0),len(x0)))
            non_zeros = np.nonzero(x0)[0]
            C0[non_zeros,non_zeros] = np.square(0.2*np.abs(x0[non_zeros]))
            C0[zeros,zeros] = scales[zeros]**2
        else:
            raise ValueError('Exact zeros in the initial value without bounds and/or scales is invalid')
    else:
        x0 = np.array(x0)
    if bounds is None:
        bounds = np.column_stack((np.repeat(-np.inf,len(x0)),np.repeat(np.inf,len(x0))))
    if np.any(x0 < bounds[:,0]) or np.any(x0 > bounds[:,1]):
        raise ValueError('Initial value outside bounds')
    if var is not None:
        if not (isinstance(var,float) or isinstance(var,np.ndarray)):
            raise ValueError('Invalid variance input, must be either 1d-array or float')

    xdim, ydim = len(x0), len(ydata) # Parameter and data space dimensions
    sd = 2.4**2 / xdim #Scaling factor for the covariance update
    # Initialize chains
    chain = np.full((samples+warmup,xdim,parallel_chains),np.nan)
    loss_chain = np.full((samples+warmup,parallel_chains),np.nan)
    
    # Initial residual vector
    res = ydata - fun(inv_transform(x0),*args).squeeze()
    
    # Variance sampling setup
    if sample_var and isinstance(var,np.ndarray):
        print('Variance sampling is ignored with vector variance input')
        sample_var = False
    if sample_var:
        var_chain = np.full((samples+warmup,parallel_chains),np.nan)
    if var is None:
        if sample_var:
            # Prior parameters for inverse gamma of variance sampling
            shape = np.sqrt(ydim)
            scale = res.std()**2 * (shape-1)
            var = (scale+0.5*ydim*res.std()**2) / (shape+0.5*ydim-1)
        else:
            spline = UnivariateSpline(np.arange(ydim),res**2,w=np.ones(ydim)/np.sqrt(ydim),s=np.std(res**2,ddof=1))
            var = spline(np.arange(ydim))
    elif var is not None and sample_var:
        shape = np.sqrt(ydim)
        scale = var * (shape-1)
        
    #Else variance is as given in the input
    if sample_var:
        var_chain[0] = var
    var = np.repeat(var,parallel_chains).reshape((-1,parallel_chains))
    
    # Prior function if given
    if log_prior is None:
        log_prior = lambda x,*args: np.zeros(parallel_chains)
        
    xt = np.repeat(x0[:,np.newaxis],parallel_chains,axis=1)
    chain[0] = xt
    Ct = np.kron(np.eye(parallel_chains,dtype=int), C0)
    res = ydata[:,np.newaxis] - fun(inv_transform(xt),*args)
    wss = np.sum(res**2/var,axis=0)
    
    # Negative log posterior (corresponds loss function assuming multivariate Gaussian likelihood)
    if len(var) == 1:
        loss = 0.5*wss + 0.5*ydim*np.log(2*np.pi*var.squeeze()) - log_prior(xt)
    else:
        loss = 0.5*wss + 0.5*np.sum(np.log(2*np.pi*var),axis=0) - log_prior(xt)
    loss_chain[0] = loss
    # Percentual progress and number of values accepted
    percent = 0
    accepted = np.zeros(parallel_chains,dtype=int)
    if progress:
        print('Warmup period...')
    for t in range(1,samples+warmup):
        if progress and (t-warmup) / (samples-1) * 100 >= percent and t >= warmup:
            print(f'{percent}%')
            percent += 10
        # t == warmup is handled separately when the warmup period ends
        #if t == warmup:
        #    continue
        #Proposed sample for each parallel chain
        proposed = np.random.multivariate_normal(xt.flatten(order='F'),Ct).reshape((xdim,parallel_chains),order='F')
        res_proposed = ydata[:,np.newaxis] - fun(inv_transform(proposed),*args)
        wss_proposed = np.sum(res_proposed**2/var,axis=0)
        if len(var) == 1:
            proposed_loss = 0.5*wss_proposed + 0.5*ydim*np.log(2*np.pi*var.squeeze()) - log_prior(proposed)
        else:
            proposed_loss = 0.5*wss_proposed + 0.5*np.sum(np.log(2*np.pi*var),axis=0) - log_prior(proposed)
        
        log_acceptance_ratios = -proposed_loss + loss
        #valid = np.all((lower[:,np.newaxis] < proposed) & (proposed < upper[:,np.newaxis]),axis=0) & ~np.isnan(wss_proposed)
        valid = np.all((bounds.min(axis=1)[:,np.newaxis] < proposed) & (proposed < bounds.max(axis=1)[:,np.newaxis]),axis=0) & ~np.isnan(wss_proposed)
        log_acceptance_ratios[~valid] = -np.inf
        acc_arr = np.log(np.random.uniform(0,1,size=parallel_chains)) <= log_acceptance_ratios
        # Update values
        if acc_arr.sum() > 0:
            xt[:,acc_arr] = proposed[:,acc_arr]
            res[:,acc_arr] = res_proposed[:,acc_arr]
            wss[acc_arr] = wss_proposed[acc_arr]
            loss[acc_arr] = proposed_loss[acc_arr]
            accepted[acc_arr] += np.ones(acc_arr.sum(),dtype=int)
            #if sample_var:
            #    var[:,acc_arr] = invgamma(shape+0.5*ydim,scale=scale+0.5*np.sum(res[:,acc_arr]**2,axis=0)).rvs(size=acc_arr.sum())
        # Update chains
        chain[t], loss_chain[t] = xt, loss
        if sample_var:
            var_chain[t] = var.flatten()
            var = invgamma(shape+0.5*ydim,scale=scale+0.5*np.sum(res**2,axis=0)).rvs(size=parallel_chains).reshape((-1,1))
        # Adaptation starts after the warmup period
        if t == warmup - 1:
            combined_samples = np.swapaxes(chain[slice(warmup//2,warmup,thinning)],1,2).reshape((warmup//2*parallel_chains//thinning,xdim))
            Ct = np.kron(np.eye(parallel_chains,dtype=int),sd*np.cov(combined_samples,rowvar=False,ddof=1))
            mean = np.repeat(combined_samples.mean(axis=0)[:,np.newaxis],parallel_chains,axis=1)
            if progress:
                print('...done')
                print('Warmup acceptance ratio:',accepted.sum()/(warmup*parallel_chains))
                print('Sampling posterior...')
            # Reset number of accepted samples    
            accepted[:] = 0
        elif t >= warmup:
            n = (t-warmup) + warmup//2 * parallel_chains // thinning
            # Recursive update for mean
            next_mean = 1/(n+1) * (n*mean + xt)
            # Recursive update for the covariance matrix
            Ct = (n-1)/n * Ct + sd/n * (n * parallel_outer_product(mean) - (n+1) * parallel_outer_product(next_mean) + parallel_outer_product(xt))
            #Update mean
            mean = next_mean
    if progress:
        print('Acceptance ratios of parallel chains:', accepted/samples)
        print('...done')
    # Combine all thinned parallel chains samples
    combined_samples = np.swapaxes(chain[warmup::thinning,:,:],1,2).reshape((N,xdim))
    losses = loss_chain[warmup::thinning,:].reshape((N,))
    if sample_var:
        variances = var_chain[warmup::thinning,:].reshape((N,))
    names = [f'x{i}' for i in range(1,xdim+1)] if names is None else names
    if sample_var:
        return xr.Dataset({'samples': xr.DataArray(combined_samples,dims=['index','params'],
                                                 coords=dict(index=range(N),params=names)),
                           'losses': xr.DataArray(losses,dims=['index'],
                                                coords=dict(index=range(N))),
                           'variances': xr.DataArray(variances,dims=['index'],
                                                   coords=dict(index=range(N)))})
    else:
        return xr.Dataset({'samples': xr.DataArray(combined_samples,dims=['index','params'],
                                                 coords=dict(index=range(N),params=names)),
                           'losses': xr.DataArray(losses,dims=['index'],
                                                coords=dict(index=range(N)))})

filename = "aci_samples_per_model.nc"
path = f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/{filename}"

start_year, end_year = 1850, 2015
params = ['log(-aci_beta)','log(aci_shape_SO2)','log(aci_shape_BC)','log(aci_shape_OC)']

if not os.path.exists(path):
    print("Sampling aerosol cloud interactions...")
    warmup = 200000
    N_samples = 2000000
    thinning = 20
    samples_xr = xr.DataArray(np.full((len(models),N_samples//thinning,len(params)),np.nan),
                              dims=['models','index','params'],
                              coords=dict(models=models,index=range(N_samples//thinning),params=params),
                              attrs={'Description': 'Samples from log-scaled posterior distributions'})
    obs_var_mean_xr = xr.DataArray(np.full(len(models),np.nan),dims=['models'],coords=dict(models=models),
                                  attrs={'Description': 'Observation variance mean'})
    obs_var_std_xr = xr.DataArray(np.full(len(models),np.nan),dims=['models'],coords=dict(models=models),
                                 attrs={'Description': 'Observation variance std'})
    for model in models:
        print(model)
        #ist = int(np.floor(years[model][0] - 1750))
        #ien = int(np.ceil(years[model][-1] - 1750))
        p0 = np.array([-1.5,1e-2,2e-2,2e-2])
        ist, ien = start_year - 1750, end_year - 1750
        bool_arr = (start_year < years[model]) & (years[model] < end_year)
        model_arr = aci[model][bool_arr]
        opt, cov = curve_fit(
            aci_log,
            [so2[ist:ien], bc[ist:ien], oc[ist:ien]],
            model_arr,
            p0 = p0,
            bounds=((-np.inf,0,0,0),(0,np.inf,np.inf,np.inf)),
            max_nfev=10000,
            method='trf'
        )
    
        pred = model_fun(opt[:,np.newaxis],so2[ist:ien],bc[ist:ien],oc[ist:ien]).squeeze()
        res = model_arr - pred
        var = np.std(res,ddof=1)**2
        x0 = transform(p0).flatten()
        C0 = np.diag([1.0**2,2.5**2,1.0**2,10.0**2])
        sampling = AM_sampling(model_fun,model_arr,N_samples,warmup=warmup,x0=x0,C0=C0,progress=True,var=var,
                               sample_var=True,thinning=thinning,log_prior=log_prior,
                               names=params,args=(so2[ist:ien],bc[ist:ien],oc[ist:ien]))
        samples_xr.loc[dict(models=model)] = sampling['samples'].data
        obs_var_mean_xr.loc[dict(models=model)] = sampling['variances'].mean(dim='index')
        obs_var_std_xr.loc[dict(models=model)] = sampling['variances'].std(ddof=1,dim='index')
    data_vars = {'samples': samples_xr,'obsvar_mean': obs_var_mean_xr,'obsvar_std':obs_var_std_xr}
    ds = xr.Dataset(data_vars=data_vars)
    ds.to_netcdf(path)
else:
    ds = Dataset(path)
    
all_samples = ds['samples'][:,:,:].data.reshape((ds['models'].size*ds['index'].size,ds['params'].size))
kde = gaussian_kde(all_samples.T,bw_method=0.15)
resampled_kde = kde.resample(prior_samples)

if plots:
    os.makedirs(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}",exist_ok=True)
    colors = {
        "CanESM5": "red",
        "CNRM-CM6-1": "orangered",
        "E3SM-2-0": "darkorange",
        "GFDL-ESM4": "yellowgreen",
        "GFDL-CM4": "yellow",
        "GISS-E2-1-G": "green",
        "HadGEM3-GC31-LL": "turquoise",
        "IPSL-CM6A-LR": "teal",
        "MIROC6": "blue",
        "MPI-ESM-1-2-HAM": "darkslateblue",
        "MRI-ESM2-0": "blueviolet",
        "NorESM2-LM": "purple",
        "UKESM1-0-LL": "crimson",
        "mean": "black",
    }
    
    
    # Plot models and corresponding fits
    fig, ax = pl.subplots(4, 4, figsize=(9.0, 4.5), squeeze=False)
    for n, model in enumerate(sorted(models, key=str.lower)):
        i,j = n // 4, n % 4
        ist, ien = start_year - 1750, end_year - 1750
        bool_arr = (start_year < years[model]) & (years[model] < end_year)
        model_arr = aci[model][bool_arr]
        ax[i,j].plot(np.arange(start_year,end_year),model_arr,color="k",ls="-",alpha=0.5,lw=1)
        model_samples = ds['samples'][models.index(model),:,:]
        preds = model_fun(inv_transform(model_samples.T),so2[ist:ien],bc[ist:ien],oc[ist:ien])
        pred_mean = preds.mean(axis=1)
        ax[i,j].plot(np.arange(start_year,end_year),pred_mean,color=colors[model],zorder=7,lw=1)
        ax[i,j].set_xlim(start_year-10, end_year+10)
        ax[i,j].set_ylim(-1.7, 0.5)
        ax[i,j].axhline(0, lw=0.5, ls=":", color="k")
        #ax[i,j].fill_between(np.arange(1850, 2015), -10, 10, color="#e0e0e0", zorder=-20)
        ax[i,j].get_xticklabels()[-1].set_ha("right")
        if j == 0:
            ax[i,j].set_ylabel("W m$^{-2}$")
        if model == "HadGEM3-GC31-LL":
            modlab = "HadGEM3"
        elif model == "MPI-ESM-1-2-HAM":
            modlab = "MPI-ESM1-2"
        else:
            modlab = model
        ax[i,j].text(0.03, 0.05, modlab, transform=ax[i, j].transAxes, fontweight="bold")
    for n in range(n+1,4*4):
        ax[n // 4, n % 4].axis("off")
    
    pl.suptitle("Aerosol-cloud interactions parameterisations")
    fig.tight_layout()
    pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/aci_calibration.png")
    pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/aci_calibration.pdf")
    pl.close()

    xlims = {'log(-aci_beta)': (-1.2,4.5), 'log(aci_shape_SO2)': (-11.0,2.0),
             'log(aci_shape_BC)': (-11.0,2.0), 'log(aci_shape_OC)': (-120.0,20.0)}
    ylims = {'log(-aci_beta)': (0.0,3.5), 'log(aci_shape_SO2)': (0.0,1.5),
             'log(aci_shape_BC)': (0.0,0.9), 'log(aci_shape_OC)': (0.0,0.1)}
    for param in params:
        fig, ax = pl.subplots(4, 4, figsize=(9.0, 4.5), squeeze=False)
        for n, model in enumerate(sorted(models, key=str.lower)):
            i,j = n // 4, n % 4
            model_param_samples = ds['samples'][models.index(model),:,params.index(param)]
            ax[i,j].hist(model_param_samples,density=True,bins=200,color=colors[model],range=xlims[param])
            ax[i,j].set_xlim(*xlims[param])
            ax[i,j].set_ylim(*ylims[param])
            if model == "HadGEM3-GC31-LL":
                modlab = "HadGEM3"
            elif model == "MPI-ESM-1-2-HAM":
                modlab = "MPI-ESM1-2"
            else:
                modlab = model
            ax[i,j].text(0.52, 0.8, modlab, transform=ax[i,j].transAxes, fontweight="bold")
        for n in range(n+1,4*4):
            ax[n // 4, n % 4].axis("off")
        pl.suptitle(f"Parameter {param} histograms per each model")
        fig.tight_layout()
        #pl.show()
        pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/{param}_samples_per_model.png")
        pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/{param}_samples_per_model.pdf")
        pl.close()
    
    for param in params:
        fig = pl.figure(figsize=(4.5, 2.25))
        ax = pl.gca()
        #all_samples = ds['samples'][:,:,params.index(param)].data.flatten()
        index = params.index(param)
        ax.hist(all_samples[:,index],density=True,bins=300,color='black',alpha=0.3,range=xlims[param],label='Combined samples')
        ax.hist(resampled_kde[index],density=True,bins=300,color='black',alpha=0.7,range=xlims[param],label='Resampled kde')
        ax.set_xlim(*xlims[param])
        ax.set_ylim(0.0,max(ylims[param])/3)
        ax.set_title(f"Combined histogram for {param} samples")
        ax.legend()
        pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/{param}_all_samples.png")
        pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/{param}_all_samples.pdf")
        pl.close()

'''
kde = gaussian_kde([np.log(n0_samp), np.log(n1_samp), np.log(n2_samp)], bw_method=0.1)
aci_sample = kde.resample(size=samples * 1, seed=63648708)
# trapezoid distribution [-2.2, -1.7, -1.0, -0.3, +0.2]
erfaci_sample = trapezoid.rvs(0.25, 0.75, size=samples, loc=-2.2, scale=2.4, random_state=71271)
'''

# Sampling with updated emissions.
df_emis_obs = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)

# overwrite RCMIP
so2 = df_emis_obs["SO2"].values
bc = df_emis_obs["BC"].values
oc = df_emis_obs["OC"].values

if plots:
    preds = model_fun(inv_transform(all_samples.T),so2,bc,oc,correction=False)
    pred_mean, pred_std = preds.mean(axis=1), preds.std(ddof=1,axis=1)
    fig = pl.figure(figsize=(4.5, 2.25))
    ax = pl.gca()
    ax.plot(df_emis_obs.index,pred_mean,color='k')
    ax.fill_between(df_emis_obs.index,pred_mean-pred_std,pred_mean+pred_std,color='k',alpha=0.3)
    ax.set_ylabel("W m$^{-2}$")
    ax.set_title('Non-corrected log(aci) ± 1σ accross all CMIP models')
    pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/log_aci_over_CMIP_models.png")
    pl.savefig(f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/log_aci_over_CMIP_models.pdf")
    
'''
beta = np.zeros(samples)
for i in tqdm(range(samples), desc="aci samples", disable=1 - progress):
    ts2010 = np.mean(
        aci_log_nocorrect(
            [so2[255:265], bc[255:265], oc[255:265]],
            1,
            np.exp(aci_sample[0, i]),
            np.exp(aci_sample[1, i]),
            np.exp(aci_sample[2, i]),
        )
    )
    ts1750 = aci_log_nocorrect(
        [so2[0], bc[0], oc[0]],
        1,
        np.exp(aci_sample[0, i]),
        np.exp(aci_sample[1, i]),
        np.exp(aci_sample[2, i]),
    )
    beta[i] = erfaci_sample[i] / (ts2010 - ts1750)
df = pd.DataFrame(
    {   "beta": beta,
        "shape_SO2": np.exp(aci_sample[0, :samples]),
        "shape_BC": np.exp(aci_sample[1, :samples]),
        "shape_OC": np.exp(aci_sample[2, :samples]),
    }
)
'''

df = pd.DataFrame(data=transform(resampled_kde).T,columns=["beta","shape_SO2","shape_BC","shape_OC"])

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_cloud.csv",
    index=False,
)
