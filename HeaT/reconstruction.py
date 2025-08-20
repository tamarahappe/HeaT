import numpy as np 
from netCDF4 import Dataset
import os
import tensorflow as tf
import argparse
import pandas as pd
from tensorflow import keras

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time
import matplotlib
import matplotlib.ticker as mticker
import warnings

import sys


HOME = "/home/thappe"
# HOME = "/scistor/ivm/the410/"

sys.path.append(os.path.expanduser(f'{HOME}/HeaT'))
from HeaT.TFrecord_utils_ERA_v2 import *
   
sys.path.append(os.path.expanduser(f"{HOME}/VAE3D/autoencoder_notebooks/LatentSpace/"))
#sys.path.append(os.path.expanduser(f"{HOME}/VAE3D/autoencoder_notebooks/LatentSpace/P1/"))
from model.autoencoder_3d_model_v1 import CVAE


#### PLOTTING FUNCTIONS
LONS= np.array([-74.53125 , -73.828125, -73.125   , -72.421875,
                   -71.71875 , -71.015625, -70.3125  , -69.609375,
                   -68.90625 , -68.203125, -67.5     , -66.796875,
                   -66.09375 , -65.390625, -64.6875  , -63.984375,
                   -63.28125 , -62.578125, -61.875   , -61.171875,
                   -60.46875 , -59.765625, -59.0625  , -58.359375,
                   -57.65625 , -56.953125, -56.25    , -55.546875,
                   -54.84375 , -54.140625, -53.4375  , -52.734375,
                   -52.03125 , -51.328125, -50.625   , -49.921875,
                   -49.21875 , -48.515625, -47.8125  , -47.109375,
                   -46.40625 , -45.703125, -45.      , -44.296875,
                   -43.59375 , -42.890625, -42.1875  , -41.484375,
                   -40.78125 , -40.078125, -39.375   , -38.671875,
                   -37.96875 , -37.265625, -36.5625  , -35.859375,
                   -35.15625 , -34.453125, -33.75    , -33.046875,
                   -32.34375 , -31.640625, -30.9375  , -30.234375,
                   -29.53125 , -28.828125, -28.125   , -27.421875,
                   -26.71875 , -26.015625, -25.3125  , -24.609375,
                   -23.90625 , -23.203125, -22.5     , -21.796875,
                   -21.09375 , -20.390625, -19.6875  , -18.984375,
                   -18.28125 , -17.578125, -16.875   , -16.171875,
                   -15.46875 , -14.765625, -14.0625  , -13.359375,
                   -12.65625 , -11.953125, -11.25    , -10.546875,
                    -9.84375 ,  -9.140625,  -8.4375  ,  -7.734375,
                    -7.03125 ,  -6.328125,  -5.625   ,  -4.921875,
                    -4.21875 ,  -3.515625,  -2.8125  ,  -2.109375,
                    -1.40625 ,  -0.703125,   0.      ,   0.703125,
                     1.40625 ,   2.109375,   2.8125  ,   3.515625,
                     4.21875 ,   4.921875,   5.625   ,   6.328125,
                     7.03125 ,   7.734375,   8.4375  ,   9.140625,
                     9.84375 ,  10.546875,  11.25    ,  11.953125,
                    12.65625 ,  13.359375,  14.0625  ,  14.765625,
                    15.46875 ,  16.171875,  16.875   ,  17.578125,
                    18.28125 ,  18.984375,  19.6875  ,  20.390625,
                    21.09375 ,  21.796875,  22.5     ,  23.203125,
                    23.90625 ,  24.609375,  25.3125  ,  26.015625,
                    26.71875 ,  27.421875,  28.125   ,  28.828125,
                    29.53125 ,  30.234375,  30.9375  ,  31.640625,
                    32.34375 ,  33.046875,  33.75    ,  34.453125,
                    35.15625 ,  35.859375,  36.5625  ,  37.265625,
                    37.96875 ,  38.671875,  39.375   ,  40.078125,
                    40.78125 ,  41.484375,  42.1875  ,  42.890625,
                    43.59375 ,  44.296875,  45.      ,  45.703125,
                    46.40625 ,  47.109375,  47.8125  ,  48.515625,
                    49.21875 ,  49.921875,  50.625   ,  51.328125,
                    52.03125 ,  52.734375,  53.4375  ,  54.140625,
                    54.84375 ,  55.546875,  56.25    ,  56.953125,
                    57.65625 ,  58.359375,  59.0625  ,  59.765625])
LATS = np.array([30.5262516 , 31.22800418, 31.92975673, 32.63150925,
                   33.33326174, 34.0350142 , 34.73676663, 35.43851902,
                   36.14027138, 36.8420237 , 37.54377599, 38.24552823,
                   38.94728044, 39.6490326 , 40.35078471, 41.05253678,
                   41.75428879, 42.45604076, 43.15779267, 43.85954452,
                   44.56129631, 45.26304804, 45.9647997 , 46.66655129,
                   47.3683028 , 48.07005424, 48.7718056 , 49.47355688,
                   50.17530806, 50.87705915, 51.57881013, 52.28056101,
                   52.98231178, 53.68406242, 54.38581295, 55.08756333,
                   55.78931357, 56.49106366, 57.19281359, 57.89456335,
                   58.59631292, 59.2980623 , 59.99981146, 60.7015604 ,
                   61.40330909, 62.10505753, 62.80680568, 63.50855352,
                   64.21030104, 64.9120482 , 65.61379497, 66.31554132,
                   67.01728721, 67.71903259, 68.42077741, 69.12252163,
                   69.82426517, 70.52600796, 71.22774993, 71.92949096,
                   72.63123095, 73.33296977, 74.03470726, 74.73644324]) 

def plot_sample(x, norm_method, title):
        
  """Plots one sample (either input or predicted) """
#   matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 40})
#   matplotlib.rc('text', usetex=True)
  fig_width = 60
  fig_height = 10
  fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height))

  if norm_method == "standardized":
    vmin, vmax = -3, 3
  elif norm_method == "standardization_cut":
    vmin, vmax = -1, 1
  elif norm_method == "difference":
    vmin, vmax = -0.5, 0.5
  else:
    vmin, vmax = 0, 1

  for t in range(5):
    stream = x[:, :, t, 0]
    psl    = x[:, :, t, 1]

    #plot stream function
    ax = plt.subplot(2, 5, t + 1, projection=ccrs.PlateCarree())
    cs = ax.pcolormesh(LONS, LATS, stream, transform=ccrs.PlateCarree(),
                       cmap="seismic", vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.set_title("Day {}".format(t))
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top    = False
    gl.xlabels_bottom = False
    gl.ylabels_right  = False
    if t == 0:
      gl.ylabels_left = True
    else:
      gl.ylabels_left = False

    #plot sea level pressure
    ax = plt.subplot(2, 5, t + 6, projection=ccrs.PlateCarree())
    cs = ax.pcolormesh(LONS, LATS, psl, transform=ccrs.PlateCarree(),
                       cmap="seismic", vmin=vmin, vmax=vmax)
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40])
    gl.xlabels_top   = False
    gl.ylabels_right = False
    if t == 0:
      gl.ylabels_left = True
    else:
      gl.ylabels_left = False    

  h_space = -0.14
  #w_space = h_space * fig_height / fig_width
  w_space = 0.03
  fig.subplots_adjust(wspace=w_space, hspace=h_space)
  fig.subplots_adjust(right=0.95)
  ax_cbar = fig.add_axes([0.96, 0.165, 0.01, 0.675])
  fig.colorbar(cs, cax=ax_cbar)

  plt.suptitle(title, fontsize=50)

  plt.show()
  plt.close()
    
#######
## RECONSTRUCTION FUNCTIONS
#######

def reconstruct_VAE_ERA5(model_file, L,  data_testbatch):
    regularizer = tf.keras.regularizers.L2(0.2)
    model = CVAE(L, filter_scaling=4)
    model.built = True
    model.load_weights(f"{model_file}")
        
    warnings.filterwarnings("ignore")
    
#     for element in data_testbatch.as_numpy_iterator(): #to convert to numpy elements
#     array = np.transpose(element[0])
#     stream_in = array[0, :, :, :] 
#     psl_in = array[1, :, :, :]
#     date = str(element[1]) + "_" + str(element[2]) + "_" + str(element[3]) 
    
    test_sample = np.zeros((1, 192, 64, 5, 2))
    for element in data_testbatch.as_numpy_iterator():
        test_sample[0] = element[0]
        test_sample_transposed = np.transpose(element[0], (1, 0, 2, 3))
    plot_sample(test_sample_transposed, "standardization_cut", "True Sample")
    
    
    
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predicted = model.decode(z, apply_sigmoid=False)
    predicted = predicted[0,:,:,:,:]
    predicted_transposed = np.transpose(predicted, (1, 0, 2, 3))
    plot_sample(predicted_transposed, "standardization_cut", "Predicted Sample")


def get_r2_reconstructed_ERA5(model_file, L,  data_parsed):
    from sklearn.metrics import r2_score
    
    regularizer = tf.keras.regularizers.L2(0.2)
    model = CVAE(L, filter_scaling=4)
    model.built = True
    model.load_weights(f"{model_file}")
        
    warnings.filterwarnings("ignore")

    
    test_sample = np.zeros((1, 192, 64, 5, 2))
    
    r2_totals = []
    
    for element in data_parsed.as_numpy_iterator():
        test_sample[0] = element[0]
        test_sample_transposed = np.transpose(element[0], (1, 0, 2, 3))    
        
        mean, logvar = model.encode(test_sample)
        z = model.reparameterize(mean, logvar)
        predicted = model.decode(z, apply_sigmoid=False)
        predicted = predicted[0,:,:,:,:]
        predicted_transposed = np.transpose(predicted, (1, 0, 2, 3))
        
        r2 = r2_score(test_sample_transposed.flatten(), predicted_transposed.flatten())
        r2_totals.append(r2)

    return r2_totals


def get_MSE_reconstructed_ERA5(model_file, L,  data_parsed):
    from sklearn.metrics import mean_squared_error
    
    regularizer = tf.keras.regularizers.L2(0.2)
    model = CVAE(L, filter_scaling=4)
    model.built = True
    model.load_weights(f"{model_file}")
        
    warnings.filterwarnings("ignore")

    
    test_sample = np.zeros((1, 192, 64, 5, 2))
    
    mse_totals = []
    
    for element in data_parsed.as_numpy_iterator():
        test_sample[0] = element[0]
        test_sample_transposed = np.transpose(element[0], (1, 0, 2, 3))    
        
        mean, logvar = model.encode(test_sample)
        z = model.reparameterize(mean, logvar)
        predicted = model.decode(z, apply_sigmoid=False)
        predicted = predicted[0,:,:,:,:]
        predicted_transposed = np.transpose(predicted, (1, 0, 2, 3))
        
        mse = mean_squared_error(test_sample_transposed.flatten(), predicted_transposed.flatten())
        mse_totals.append(mse)

    return mse_totals

def get_heatwave_means(model_file, L, data_parsed):
    
    regularizer = tf.keras.regularizers.L2(0.2)
    model = CVAE(L, filter_scaling=4)
    model.built = True
    model.load_weights(f"{model_file}")
        
    warnings.filterwarnings("ignore")

    
    test_sample = np.zeros((1, 192, 64, 5, 2))
    
    heatwave_means = []
    
    for element in data_parsed.as_numpy_iterator():
        test_sample[0] = element[0]
        test_sample_transposed = np.transpose(element[0], (1, 0, 2, 3))    
        mean, logvar = model.encode(test_sample)
        heatwave_means.append(mean.numpy()[0])

    return heatwave_means