import numpy as np
import xarray as xr

# Approximate Partial Radiation Perturbation method (Taylor et al., 2007)
# Inputs:
#   filenames1:  list of files to the netCDF output for time period 1
#   filenames2:  list of files to the netCDF output for time period 2
#   time_range1:  e.g., slice(0,12)
#   time_range2:  e.g., slice(0,12)
#   vnames_aux: coordinate variables that you want keep, e.g., ['lat', 'lon', 'area']
# Outputs: saved netcdf file

# Author: Rick Russotto and Yen-Ting Hwang, 2017, https://github.com/rdrussotto/pyAPRP
# Author: Jiang Zhu, Xarray implementations and CESM friendly, 2024


# main program
def aprp_cesm(filenames1, time_range1,
              filenames2, time_range2, vnames_aux, out_file):

    # Load files and compute derived variables and parameters
    ds1 = load_prepare_data(filenames1, time_range1, vnames_aux)
    ds1 = compute_parameters(ds1)

    ds2 = load_prepare_data(filenames2, time_range2, vnames_aux)
    ds2 = compute_parameters(ds2)

    ds1['time'] = ds2['time']

    # Do the calculation and save netcdf file
    ds = compute_d_albedo(ds1, ds2, vnames_aux)
    save_ds_netcdf(ds, out_file)


# function to load data
def load_prepare_data(filenames, time_range, vnames_aux):

    ds = xr.open_mfdataset(filenames, parallel=True,
                           data_vars='minimal',
                           coords='minimal',
                           compat='override',
                           chunks={'time': 12}).isel(time=time_range)

    rsds = ds.FSDS
    rsut = ds.FSUTOA
    rsdscs = ds.FSDSC
    clt = ds.CLDTOT
    ts = ds.TS

    rsus = ds.FSDS - ds.FSNS
    rsdt = ds.FSUTOA + ds.FSNTOA  # down = net + up
    rsuscs = ds.FSDSC - ds.FSNSC
    rsutcs = rsdt - ds.FSNTOAC  # Downward SW at TOA should be same regardless of clouds.

    # Calculate the overcast versions of rsds, rsus, rsut from the clear-sky and all-sky data
    # Can derive this algebraically from Taylor et al., 2007, Eq. 3
    clt_crit = 1e-5
    rsdsoc = xr.where(
        clt > clt_crit, (rsds-(1.-clt)*(rsdscs))/clt, 0.0)
    rsusoc = xr.where(
        clt > clt_crit, (rsus-(1.-clt)*(rsuscs))/clt, 0.0)
    rsutoc = xr.where(
        clt > clt_crit, (rsut-(1.-clt)*(rsutcs))/clt, 0.0)

    ds['ts'] = ts
    ds['clt'] = clt
    ds['rsds'] = rsds
    ds['rsus'] = rsus
    ds['rsut'] = rsut
    ds['rsdt'] = rsdt
    ds['rsutcs'] = rsutcs
    ds['rsdscs'] = rsdscs
    ds['rsuscs'] = rsuscs
    ds['rsdsoc'] = rsdsoc
    ds['rsusoc'] = rsusoc
    ds['rsutoc'] = rsutoc

    return ds[['ts', 'clt', 'rsds', 'rsus',
               'rsut', 'rsdt', 'rsutcs', 'rsdscs',
               'rsuscs', 'rsdsoc', 'rsusoc', 'rsutoc']+vnames_aux]


# function to compute parameters related to the idealized single-layer radiative transfer model
def compute_parameters(ds):

    # clearsky part
    # Surface albedo
    a_clr = ds['rsuscs']/ds['rsdscs']

    # Ratio of incident surface flux to insolation
    Q = ds['rsdscs']/ds['rsdt']

    # Atmospheric transmittance (Eq. 9)
    mu_clr = ds['rsutcs']/ds['rsdt']+Q*(1.-a_clr)

    # Atmospheric scattering coefficient (Eq. 10)
    ga_clr = (mu_clr-Q)/(mu_clr-a_clr*Q)

    # Overcast parameters
    # Surface albedo
    a_oc = ds['rsusoc']/ds['rsdsoc']

    # Ratio of incident surface flux to insolation
    Q = ds['rsdsoc']/ds['rsdt']

    # Atmospheric transmittance (Eq. 9)
    mu_oc = ds['rsutoc']/ds['rsdt']+Q*(1.-a_oc)

    # Atmospheric scattering coefficient (Eq. 10)
    ga_oc = (mu_oc-Q)/(mu_oc-a_oc*Q)

    # Calculating cloudy parameters based on clear-sky and overcast ones
    # Difference between _cld and _oc: _cld is due to the cloud itself, as opposed to
    # scattering and absorption from all constituents including clouds in overcast skies.
    mu_cld = mu_oc / mu_clr  # Eq. 14
    ga_cld = (ga_oc-1.)/(1.-ga_clr)+1.  # Eq. 13

    # Save the relevant variables for later use
    ds['a_clr'] = a_clr
    ds['a_oc'] = a_oc
    ds['mu_clr'] = mu_clr
    ds['mu_cld'] = mu_cld
    ds['ga_clr'] = ga_clr
    ds['ga_cld'] = ga_cld

    return ds


# function to perform the APRP calculation
def compute_d_albedo(ds1, ds2, vnames_aux):
    clt1 = ds1['clt']
    clt2 = ds2['clt']
    a_oc1 = ds1['a_oc']
    a_oc2 = ds2['a_oc']
    a_clr1 = ds1['a_clr']
    a_clr2 = ds2['a_clr']

    mu_cld1 = ds1['mu_cld']
    mu_cld2 = ds2['mu_cld']
    ga_cld1 = ds1['ga_cld']
    ga_cld2 = ds2['ga_cld']

    mu_clr1 = ds1['mu_clr']
    mu_clr2 = ds2['mu_clr']
    ga_clr1 = ds1['ga_clr']
    ga_clr2 = ds2['ga_clr']

    # Base state albedo
    A1 = compute_albedo(clt1, a_clr1, a_oc1, mu_clr1,
                        mu_cld1, ga_clr1, ga_cld1)
    A2 = compute_albedo(clt2, a_clr2, a_oc2, mu_clr2,
                        mu_cld2, ga_clr2, ga_cld2)

    # Change in albedo due to each component (Taylor et al., 2007, Eq. 12b)
    dA_c = .5*(compute_albedo(clt2, a_clr1, a_oc1, mu_clr1, mu_cld1, ga_clr1, ga_cld1)-A1) + \
        .5*(A2-compute_albedo(clt1, a_clr2, a_oc2, mu_clr2, mu_cld2, ga_clr2, ga_cld2))

    dA_a_clr = .5*(compute_albedo(clt1, a_clr2, a_oc1, mu_clr1, mu_cld1, ga_clr1, ga_cld1)-A1) + \
            .5*(A2-compute_albedo(clt2, a_clr1, a_oc2, mu_clr2, mu_cld2, ga_clr2, ga_cld2))

    dA_a_oc = .5*(compute_albedo(clt1, a_clr1, a_oc2, mu_clr1, mu_cld1, ga_clr1, ga_cld1)-A1)+ \
           .5*(A2-compute_albedo(clt2, a_clr2, a_oc1, mu_clr2, mu_cld2, ga_clr2, ga_cld2))

    dA_mu_clr = .5*(compute_albedo(clt1, a_clr1, a_oc1, mu_clr2, mu_cld1, ga_clr1, ga_cld1)-A1)+\
             .5*(A2-compute_albedo(clt2, a_clr2, a_oc2, mu_clr1, mu_cld2, ga_clr2, ga_cld2))

    dA_mu_cld = .5*(compute_albedo(clt1, a_clr1, a_oc1, mu_clr1, mu_cld2, ga_clr1, ga_cld1)-A1)+\
             .5*(A2-compute_albedo(clt2, a_clr2, a_oc2, mu_clr2, mu_cld1, ga_clr2, ga_cld2))

    dA_ga_clr = .5*(compute_albedo(clt1, a_clr1, a_oc1, mu_clr1, mu_cld1, ga_clr2, ga_cld1)-A1)+\
             .5*(A2-compute_albedo(clt2, a_clr2, a_oc2, mu_clr2, mu_cld2, ga_clr1, ga_cld2))

    dA_ga_cld = .5*(compute_albedo(clt1, a_clr1, a_oc1, mu_clr1, mu_cld1, ga_clr1, ga_cld2)-A1)+\
            .5*( A2-compute_albedo(clt2, a_clr2, a_oc2, mu_clr2, mu_cld2, ga_clr2, ga_cld1))

    # Combine different components into changes due to surface albedo, atmospheric clear-sky and atmospheric cloudy-sky
    dA_a = dA_a_clr + dA_a_oc  # Eq. 16a
    dA_cld = dA_mu_cld + dA_ga_cld + dA_c  # Eq. 16b
    dA_clr = dA_mu_clr + dA_ga_clr  # Eq. 16c

   # Calculate radiative effects in W/m^2 by multiplying negative of planetary albedo changes by downward SW radation
    # (This means positive changes mean more downward SW absorbed)
    surface = -dA_a*ds2['rsdt']  # Radiative effect of surface albedo changes
    cloud = -dA_cld*ds2['rsdt']  # Radiative effect of cloud changes

    # Radiative effect of non-cloud SW changes (e.g. SW absorption)
    noncloud = -dA_clr*ds2['rsdt']

    # Broken down further into the individual terms in Eq. 16
    # Effects of surface albedo in clear-sky conditions
    surface_clr = -dA_a_clr*ds2['rsdt']
    # Effects of surface albedo in overcast conditions
    surface_oc = -dA_a_oc*ds2['rsdt']

    cloud_c = -dA_c*ds2['rsdt']  # Effects of changes in cloud fraction
    # Effects of atmospheric scattering in cloudy conditions
    cloud_ga = -dA_ga_cld*ds2['rsdt']
    # Effects of atmospheric absorption in cloudy conditions
    cloud_mu = -dA_mu_cld*ds2['rsdt']

    # Effects of atmospheric scattering in clear-sky conditions
    noncloud_ga = -dA_ga_clr*ds2['rsdt']
    # Effects of atmospheric absorption in clear-sky conditions
    noncloud_mu = -dA_mu_clr*ds2['rsdt']

    # Calculate more useful radiation output
    # Change in cloud radiative effect
    CRF = ds1['rsut'] - ds1['rsutcs'] - ds2['rsut'] + ds2['rsutcs']
    # Change in clear-sky upward SW flux at TOA
    cs = ds1['rsutcs'] - ds2['rsutcs']

    # Return all the variables calculated here

    ds = ds1.copy(deep=True)
    ds['ts1'] = ds1.ts
    ds['ts2'] = ds2.ts
    ds['A1'] = A1
    ds['A2'] = A2
    ds['dA_c'] = dA_c
    ds['dA_a_clr'] = dA_a_clr
    ds['dA_a_oc'] = dA_a_oc
    ds['dA_mu_clr'] = dA_mu_clr
    ds['dA_mu_cld'] = dA_mu_cld
    ds['dA_ga_clr'] = dA_ga_clr
    ds['dA_ga_cld'] = dA_ga_cld
    ds['dA_a'] = dA_a
    ds['dA_cld'] = dA_cld
    ds['dA_clr'] = dA_clr
    ds['surface'] = surface
    ds['cloud'] = cloud
    ds['noncloud'] = noncloud
    ds['surface_clr'] = surface_clr
    ds['surface_oc'] = surface_oc
    ds['cloud_c'] = cloud_c
    ds['cloud_ga'] = cloud_ga
    ds['cloud_mu'] = cloud_mu
    ds['noncloud_ga'] = noncloud_ga
    ds['noncloud_mu'] = noncloud_mu
    ds['CRF'] = CRF
    ds['cs'] = cs
    ds = ds.fillna(0)

    ds = ds[['ts1', 'ts2', 'A1', 'A2', 'dA_c', 'dA_a_clr', 'dA_a_oc',
             'dA_mu_clr', 'dA_mu_cld', 'dA_ga_clr', 'dA_ga_cld', 'dA_a',
             'dA_cld', 'dA_clr', 'surface', 'cloud', 'noncloud',
             'surface_clr', 'surface_oc', 'cloud_c', 'cloud_ga', 'cloud_mu',
             'noncloud_ga', 'noncloud_mu', 'CRF', 'cs'] + vnames_aux]

    return ds


# Function to calculate the planetary albedo, A.
# Inputs: (see Fig. 1 of Taylor et al., 2007)
#   c: fraction of the region occupied by clouds
#   a_clr: clear sky surface albedo (SW flux up / SW flux down)
#   a_oc: overcast surface albedo
#   mu_clr: clear-sky transmittance of SW radiation
#   mu_cld: cloudy-sky transmittance of SW radiation
#   ga_clr: clear-sky atmospheric scattering coefficient
#   ga_cld: cloudy-sky atmospheric scattering coefficient
# Labeled with equation numbers from Taylor et al. 2007
def compute_albedo(c, a_clr, a_oc, mu_clr, mu_cld, ga_clr, ga_cld):
    mu_oc = mu_clr*mu_cld  # Eq. 14
    ga_oc = 1. - (1.-ga_clr)*(1.-ga_cld)  # Eq. 13
    A_clr = mu_clr*ga_clr + mu_clr*a_clr * \
        (1.-ga_clr)*(1.-ga_clr)/(1.-a_clr*ga_clr)  # Eq. 7 (clear-sky)
    A_oc = mu_oc*ga_oc + mu_oc*a_oc * \
        (1.-ga_oc)*(1.-ga_oc)/(1.-a_oc*ga_oc)  # Eq. 7 (overcast sky)
    A = (1-c)*A_clr + c*A_oc  # Eq. 15
    return A


# Function to save netcdf
def save_ds_netcdf(ds, filename):
    enc_dv = {xname: {'_FillValue': None} for xname in ds.data_vars}
    enc_c = {xname: {'_FillValue': None} for xname in ds.coords}
    enc = {**enc_c, **enc_dv}

    ds.to_netcdf(filename,  unlimited_dims='time', encoding=enc)