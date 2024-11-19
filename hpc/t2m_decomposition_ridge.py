import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.linear_model import Ridge
import pandas as pd
import os

os.chdir("/home/thappe/HeaT")

from HeaT.dyn_decomp import *

if __name__ == "__main__":

    data_path = "/home/thappe/data/"

    ## LOAD DATA

    T2M_file = "1940-2023_T2M_westEU_JJA.nc"
    t2m = xr.open_dataset(f"{data_path}{T2M_file}")["t2m"] - 273.15

    STREAM500_file = "STREAM500_era5_ExtendedWestEU_0.25degr_1940_2022_JJA.nc"
    stream500 = xr.open_dataset(f"{data_path}{STREAM500_file}")["stream"]

    yearly_GMST = pd.read_csv(f"{data_path}NASA_GMST.csv", delimiter=";", skiprows=1, index_col=0)

    ## SET GLOBAL VARIABLES
    lower_year=1940 
    upper_year=2022
    window=20

    ## NOT CROSS VALIDATING FOR ALL GRIDPOINTS
    cross_validate=False
    alpha=10

    ## assert the stream500 is large enough for the set window size, given t2m
    lat_extent_exeeded = (max(t2m.latitude)+window > max(stream500.latitude)) or (min(t2m.latitude)-window > min(stream500.latitude))
    lon_extent_exeeded = (max(t2m.longitude)+window > max(stream500.longitude)) or (min(t2m.longitude)-window > min(stream500.longitude))

    if lat_extent_exeeded or lon_extent_exeeded:
        print("stream500 extent is not enough for the window size given.")
        print(f"Stream500 extent is lat:{(int(min(stream500.latitude))), int(max(stream500.latitude))}, lon:{int(min(stream500.longitude)), int(max(stream500.longitude))}")
        print(f"t2m extent is lat:{(int(min(t2m.latitude))), int(max(t2m.latitude))}, lon:{int(min(t2m.longitude)), int(max(t2m.longitude))}")
        print(f"window is {window}")
        raise ValueError(f'lat or lon extent is exceeded: Lat_exeeded={lat_extent_exeeded} and Lon_exeeded={lon_extent_exeeded}')


    #EMPTY ARRAYS IN THE SAME SHAPE AS T2M? 
    ntime= (upper_year - lower_year + 1) * 92 

    optimal_lambda_values = np.zeros((t2m.shape[1],t2m.shape[2]))
    MSE_values = np.zeros((t2m.shape[1],t2m.shape[2]))
    R2_values = np.zeros((t2m.shape[1],t2m.shape[2]))
    T2M_thermodynamic_std_values = np.zeros((ntime,t2m.shape[1],t2m.shape[2]))
    T2M_dynamical_std_values = np.zeros((ntime,t2m.shape[1],t2m.shape[2]))
    T2M_dynamic_values =np.zeros((ntime,t2m.shape[1],t2m.shape[2]))
    T2M_combined_values = np.zeros((ntime,t2m.shape[1],t2m.shape[2]))

    for i, lat in enumerate(t2m.latitude):
        for j, lon in enumerate(t2m.longitude):
            print(j)
            if j>2:
                break
            ## first process the data correctly
            Y_, X_, penalty, t2m_mean, t2m_std, stream500_x2, GMST_x1 = preprocess_data(
                lon, lat, t2m, stream500, yearly_GMST,
                lower_year=lower_year, upper_year=upper_year, window=window)
            ## fit the model for this gridpoint
            optimal_lambda, MSE, R2, T2M_thermodynamic_std, T2M_dynamical_std, T2M_dynamic, T2M_combined = decompose_one_gridpoint(
                Y_, X_, penalty, t2m_mean, t2m_std, stream500_x2, GMST_x1,
                cross_validate = cross_validate, alpha = alpha)

            ## save the data for this gridpoint
            optimal_lambda_values[i, j] = optimal_lambda #this will be nans if cross_validate = False 
            MSE_values[i, j] = MSE
            R2_values[i, j] = R2
            T2M_thermodynamic_std_values[:, i, j] = T2M_thermodynamic_std
            T2M_dynamical_std_values[:, i, j] = T2M_dynamical_std
            T2M_dynamic_values[:, i, j] = T2M_dynamic
            T2M_combined_values[:, i, j] = T2M_combined

    #### SAVING INFO TO DATASET AND NC

    t2m_sliced = t2m.sel(time=t2m.time.dt.year>= lower_year)
    t2m_sliced = t2m_sliced.sel(time=t2m_sliced.time.dt.year<= upper_year)

    filename = f'T2M_westEU_decomposed_{lower_year}-{upper_year}_window={window}_CV={cross_validate}_alpha={alpha}_test.nc'
    
    ## data coordinates are (time, latitude, longitude) for the variables: lambda, MSE, R2, T2M_thermo, etc... 
    description=f"Ridge regression to decompose T2M into thermo and dynamical components. Cross_validation={cross_validate}, alpha={alpha}. Model information is stored as well. Standardized thermo and dynamical components as predicted are saved. T2M-dynamic and T2M-combined are reconstructed using STD and MEAN per gridpoint. "

    ds_out = xr.Dataset(
        data_vars=dict(
            optimal_lambda=(["latitude", "longitude"], optimal_lambda_values),
            MSE=(["latitude", "longitude"],MSE_values),
            R2=(["latitude", "longitude"],R2_values),
            T2M_thermodynamic_std=(["time", "latitude", "longitude"],T2M_thermodynamic_std_values),
            T2M_dynamical_std=(["time", "latitude", "longitude"],T2M_dynamical_std_values),
            T2M_dynamic=(["time", "latitude", "longitude"],T2M_dynamic_values),
            T2M_combined=(["time", "latitude", "longitude"],T2M_combined_values),
        ),
        coords=dict(
            longitude=("longitude", t2m_sliced.longitude.values),
            latitude=("latitude", t2m_sliced.latitude.values),
            time=t2m_sliced.time.values,
    #         reference_time=reference_time,
        ),
        attrs=dict(description=description),
    )
    
    ds_out.to_netcdf(f"{data_path}/T2M_decomposition/{filename}")

