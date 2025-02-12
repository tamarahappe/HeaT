import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os

def multiyear_multiday_running_percentile(data:xr.DataArray, 
                                          percentile:float=90, day_window:int=15, 
                                          filename_out=""):
    """
    Takes in data, calculates multi-year multi-day running Nth-percentile, centered. 
    """
    n_summerdays = data.sel(time=data.time.dt.year==int(data.time.dt.year[0])).shape[0]
    day_window_half = int((day_window-1)/2)
    n_years = np.unique(data.time.dt.year).shape[0]

    percentile_array = np.zeros((n_summerdays-day_window, data.shape[1], data.shape[2]))

    for day in np.arange(n_summerdays-day_window): #number of summer days to loop over
    #     print("day is:", day)
        day_values_of_all_years = np.zeros((day_window*n_years, data.shape[1], data.shape[2]))
        summerday = day + day_window_half
        for i in range(n_years):
            #to get the n-window surrounding summerday of interest, for all years
            #lower day is summerday (day over which we loop + )
            lower_day= summerday-day_window_half+(i*n_summerdays)
            upper_day = 1+summerday+day_window_half+(i*n_summerdays)
    #         (print(lower_day, upper_day))
            year_window = data[lower_day:upper_day,:,:]
            day_values_of_all_years[i*day_window:i*day_window+day_window,:,:]=year_window
            
        percentile_of_day = np.nanpercentile(day_values_of_all_years, percentile, axis=(0))    
        percentile_array[day, :,:] = percentile_of_day
    
    ## now create xarray
    
    #this needs to be different, based on nsummerdays and day_window
    time = data.sel(time=data.time.dt.year==int(data.time.dt.year[0])).time
    time_shortened = time[day_window_half+1:-day_window_half]
    
    ds_out = xr.Dataset(
                    data_vars=dict(
                percentile=(["time", "latitude", "longitude"], percentile_array),),
                        coords=dict(
                longitude=("longitude", data.longitude.values),
                latitude=("latitude", data.latitude.values),
                time=time_shortened.values,
            ),
            attrs=dict(description=f"{day_window}-day multi-year running {percentile}th percentile, centered. Over {n_years} years."),
        )

    ds_out.to_netcdf(f"{filename_out}")
    
    return ds_out

    
    