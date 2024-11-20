import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.linear_model import Ridge
import pandas as pd
import sys

data_path = "/home/thappe/data/"

sys.path.append("/home/thappe/HeaT")
from HeaT.custom_enet import CustomENet, CustomENetCV
# from scipy.optimize import minimize_scalar
# from sklearn.linear_model import ElasticNetCV
# from sklearn.preprocessing import StandardScaler
# from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

def preprocess_data(lon, lat, 
                     t2m, stream500, yearly_GMST,
                  lower_year=1940, upper_year=2022,
                 window=20, LENTIS_GRID=True):
    """
    Takes input data and selects the correct time-period, appropiate (lon,lat) and window.
    It prepares the GMST yearly data to reshape to adjust to the timeperiod of the temperature data.
    It also standardizes the input and output data.
    It concatenates the x_input data into one array
    LENTIS_GRID parameter is important for accessing the correct t2m lat/lon values
    
    it returns Y_, X_ ; the data for the regression model
    it returns t2m_mean and t2m_std; the gridpoint statistics
    """
    ##########################
    ### CUT AND SLICE DATA ###
    ##########################
    lon_bounds = (lon-window, lon+window)
    lat_bounds = (lat-window, lat+window)
    
    ###t2m###
    t2m_cut = t2m.sel(time=t2m.time.dt.year>= lower_year)
    t2m_cut= t2m_cut.sel(time=t2m_cut.time.dt.year<= upper_year)
    
    if LENTIS_GRID:
        t2m_cut = t2m_cut.sel(lon=lon, lat=lat)
    elif not LENTIS_GRID:
        t2m_cut = t2m_cut.sel(longitude=lon, latitude=lat)

    ###GMST###
    ny = upper_year - lower_year + 1 
    LOWESS5y_yearly_GMST = yearly_GMST["Lowess(5)"][-ny:].to_numpy() ## select correct timeperiod
    GMST_reshaped = np.empty(92*ny,) #empty array with shape of number of days (e.g 92 per summer, per year)
    for i, GMST in enumerate(LOWESS5y_yearly_GMST): ## loop over the GMST data to fill the array
        year_array = np.array(92 * [GMST], dtype=float)
        GMST_reshaped[i*92:(i+1)*92] = year_array[:]
    #create xarray 
    GMST_xr = xr.DataArray(
        data=GMST_reshaped,
        dims=["time"],
        coords=dict(
            time=t2m_cut.time.dt.year,),
        attrs=dict(
            description="Global Mean Surface Temperature anomalie from NASA, 5 year LOWESS",
            units="degC",),)
    ###stream500###
    stream500_cut = stream500.sel(
        longitude=slice(lon_bounds[0], lon_bounds[1]), latitude=slice(lat_bounds[1], lat_bounds[0]))
    stream500_cut = stream500_cut.sel(time=stream500_cut.time.dt.year>=lower_year)
    stream500_cut = stream500_cut.sel(time=stream500_cut.time.dt.year<=upper_year)
    
    ##########################
    ###  STANDARDIZE DATA  ###
    ##########################
    ## t2m
    t2m_y = (t2m_cut - t2m_cut.mean()) / t2m_cut.std()
    ## GMST
    GMST_x1 = (GMST_xr - GMST_xr.mean()) / GMST_xr.std()
    ## stream
    stream500_mean = stream500_cut.mean(dim="time")
    stream500_std = stream500_cut.std(dim="time")
    stream500_x2 = (stream500_cut - stream500_mean) /stream500_std
    
    #############################
    ### RESHAPE + CONCAT DATA ###
    #############################
    labels = []
    values = []
    dict_vars = {"GMST":GMST_x1, "STREAM500":stream500_x2}
    time=stream500_x2.time

    #empty array
    values_concated = np.empty((time.shape[0], 1+(stream500_x2.shape[1]*stream500_x2.shape[2])))
    labels_concated = np.empty((1+(stream500_x2.shape[1]*stream500_x2.shape[2])), dtype="U8")
    
    labels_concated[0]="GMST" 
    labels_concated[1:]="STREAM500"
    
    for i, timestep in enumerate(time):  
        values_timestep = []
        for var in dict_vars.keys():
            y = dict_vars[var].values[i] #values for this specific timestep
            if var == "GMST":
                values_timestep.append(y)
            if var == "STREAM500":
                y = y.reshape(y.shape[0]*y.shape[1])
                for y_val in y:
                    values_timestep.append(y_val)
        #put in array     
        values_concated[i]=np.array(values_timestep)[:]
    
    X_concated = xr.DataArray(values_concated, 
             dims=['time','feature'], 
            coords=dict(
                time=time, 
                feature=labels_concated))
    X_ = X_concated.values.astype("float32")
    
    Y_ = t2m_y.values.reshape(7636, 1).astype("float64")
    
    penalty = np.ones(X_.shape[1]) 
    penalty[np.where(labels_concated=='GMST')] = 0 
    
    return Y_, X_, penalty, t2m_cut.mean(), t2m_cut.std(), stream500_x2, GMST_x1

def decompose_one_gridpoint(Y_, X_, penalty, 
                            t2m_mean, t2m_std,
                            stream500_x2, GMST_x1,
                            cross_validate = False,
                            alpha = 10,
                            K=5
                            ):
    """
    Takes prepared data from _preprocess_data() and fits the ridge model for one gridpoint.
    Takes t2m_mean and std values to calculate the non-standardize t2m data  
    
    if cross_validate == True: K determines amount of folds for cross validation. 
    if cross_validate == False: alpha is needed --> optimal_lambda will be NaN
    
    returns: optimal_lambda, MSE, R2, t2m_thermodynamic_std, t2m_thermodynamic, t2m_dynamic_std, t2m_dynamic
    
    """
    
    #set cross validation and lambda (alphas) values 
    if cross_validate == True:
        cv = RepeatedKFold(n_splits=K, n_repeats=1, random_state=2023) 
        alphas = np.geomspace(0.01, 10, num=100) #penalty for regularization (also called lambda)

        #fit the model
        model_ce = CustomENetCV(cv, l1_ratio=0, 
                                standardize=False, #already standardized
                                fit_intercept=False, #y is centered (standadized) 
                                refit=True, #to refit the last model with the best alpha value 
                                alphas=alphas, 
                                tol=1e-4, #dec of coefficients
                                max_iter=5000, #needs 5000 misschien minder maar even om de code te testen
                                random_state=2023)
        model_ce.fit(X_, Y_, penalty)
  
            
        #Grab optimal lambda from the fitted model (based on minimum MSE)
        optimal_lambda = model_ce.alpha_best 

        
        
    elif cross_validate == False:
        
        model_ce = CustomENet(l1_ratio=0, 
                        standardize=False, #already standardized
                        fit_intercept=False, #y is centered (standadized) 
                        alpha=alpha, #just to try  
                        tol=1e-4, #dec of coefficients
                        max_iter=5000, #needs 5000 misschien minder maar even om de code te testen
                        random_state=2023)
        
        model_ce.fit(X_, Y_, penalty)
        optimal_lambda = np.NaN
    
    #Predict and calculate validation scores 
    y_pred = model_ce.predict(X_).reshape(-1, 1)
    MSE = mean_squared_error(Y_, y_pred, squared=False)
    R2 = r2_score(Y_, y_pred)

    
    #Grab weights from the model 
    weights = model_ce.w
    GMSTw = weights[0]
    stream_weigths_reshaped = weights[1:].reshape(stream500_x2.shape[1], stream500_x2.shape[2]) 
    

    #### predict dynamic and thermodynamic parts
    T2M_dynamical_std = (stream_weigths_reshaped * stream500_x2).sum(dim=("longitude", "latitude"))
    T2M_dynamical = (T2M_dynamical_std * t2m_std ) + t2m_mean  #convert to T2M values
    
    T2M_thermodynamical_std = (GMSTw * GMST_x1)
    
    T2M_combined = (T2M_thermodynamical_std * t2m_std).values + T2M_dynamical.values #make sure to only add the mean value once 

    
    return optimal_lambda, MSE, R2, T2M_thermodynamical_std.values, T2M_dynamical_std.values, T2M_dynamical.values, T2M_combined

   




   



   




   



