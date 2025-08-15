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
import xarray as xr

import pickle 
from sklearn.metrics.pairwise import pairwise_distances_argmin, cosine_similarity


# HOME = "/home/thappe/" #snellius
# os.chdir(f"{HOME}HeaT")
sys.path.append(os.path.expanduser('/home/thappe/HeaT'))

from HeaT.clustering import Clustering



        