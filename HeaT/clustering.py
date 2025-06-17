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

import pickle 
from sklearn.metrics.pairwise import pairwise_distances_argmin, cosine_similarity


# HOME = "/home/thappe/" #snellius
# os.chdir(f"{HOME}HeaT")
sys.path.append(os.path.expanduser('/home/thappe/HeaT'))

from HeaT.TFrecord_utils_ERA_v2 import *


class Clustering:
    def __init__(self, t2m_name, VAE_model, data_path, 
                 cluster_model_name="GMM", cluster_type="fit", cluster_k=4):
        
        assert t2m_name in ["t2m", "t2m_dynamic", "t2m_minus_thermo"], "data not available for this var"
        self.t2m_name = t2m_name
        
        self.VAE_model = VAE_model
        self.data_path = data_path
        self.cluster_model_name = cluster_model_name
        self.cluster_type = cluster_type
        self.cluster_k = cluster_k #default is 4, nr of clusters
        
        self.heatwave_means = None
        self.heatwave_dates = None
        self.tf_records = None 
        
        self.cluster_model = None 
        
        self.y_pred = None
        self.y_pred_proba = None 
        
        #load in encoded heatwave data
        self.load_encoded_heatwaves()
        
        #load cluster model and do clustering
        self.load_cluster_model()
        self.clustering()
        
        #load tf_records data
        self.load_tf_records()
        
        
        
    def load_encoded_heatwaves(self):
        """ 
        To load in the ERA5 encoded heatwave means and associated datetime for given t2m name. 
        """
        heatwave_means = pd.read_csv(f'{self.data_path}/ERA5_{self.t2m_name}_encoded_heatwaves_L{self.VAE_model[0]}.csv', 
                             skiprows=1, header=None)
        heatwave_dates = pd.read_csv(f'{self.data_path}/ERA5_{self.t2m_name}_heatwaves_dates.csv',
                                     skiprows=1, header=None)
        
        #convert into dates from datetime 
        heatwave_dates[0] = pd.to_datetime(heatwave_dates[0]).dt.date
        dates = heatwave_dates.values.flatten()
        
        #assign to object
        self.heatwave_means = heatwave_means
        self.heatwave_dates = dates

    def load_cluster_model(self):
        assert self.cluster_model_name in ["GMM", "Kmeanseucl"], "Model type not available to load"
        print(f"loading {self.cluster_model_name}")
        f=f'{self.data_path}/{self.cluster_model_name}.pkl'
        self.cluster_model = pickle.load(open(f, 'rb')) ## to open the model
        
                
    def load_tf_records(self):
        """
        To load in tf_records (standardized cut) from given t2m data.
        """
        
        tf_record_file = f"/home/thappe/data/tf_records/TF_record_ERA5_{self.t2m_name}_1940-2023_standardization_cut.tfrecord"
        data = load_tfrecords_ERA5(tf_record_file)
        data_parsed = data.map(parse_function_full_era5)
        self.tf_records = data_parsed
        
        
    def clustering(self):
        """clustering era5data from heatwave means with model as loaded, depending on the way of fitting"""

        assert self.cluster_type in ["fit", "transfer", "project"], "type of clustering not recognized"
     
        
        if self.cluster_type == "fit":
            
            if self.cluster_k != 4:
                print(f"changing nr of k for clustering to {self.cluster_k}")
                if self.cluster_model_name == "GMM":
                    self.cluster_model.n_components = self.cluster_k
                elif self.cluster_model_name == "Kmeanseucl":
                    self.cluster_model.n_clusters = self.cluster_k
 
            self.y_pred = self.cluster_model.fit_predict(self.heatwave_means)
        
        elif self.cluster_type == "transfer": #concat datasets and then fit 
            # https://stackoverflow.com/questions/29095769/sklearn-gmm-on-large-datasets 
            lentis_heatwave_means = pd.read_csv(f'{self.data_path}/LENTIS_Heatwave_means_VAE_L128_1000epochs.csv', 
                             skiprows=1, header=None)
            to_cluster = pd.concat([self.heatwave_means, lentis_heatwave_means])
            self.y_pred = self.cluster_model.fit_predict(to_cluster) 

        elif self.cluster_type == "project":
            self.y_pred = self.cluster_model.predict(self.heatwave_means)
            
        #if model is GMM, also get the probabilities 
        if self.cluster_model_name == "GMM":
            self.y_pred_proba = self.cluster_model.predict_proba(self.heatwave_means) 
        
        
    def plot_over_time(self, savefig=False, outpathfigure=""):
        "clustering over time, with plot"
        
        ### counting the amount of heatwaves per cluster, per year 
        years = np.arange(1940, 2024, 1)
        year_counts_for_plotting_reversed  = {}

        for cluster_id in [1,2,3,4]:
            year_counts_for_plotting_reversed[cluster_id] = np.zeros_like(years)

        for date, cluster in zip(self.heatwave_dates, self.y_pred):
            cluster_id = cluster+1
            year_index = np.where(years==date.year)
            year_counts_for_plotting_reversed[cluster_id][year_index] += 1 #add one to this year and cluster count
        
        #### plotting 
        fig, axes = plt.subplots(2,2, figsize=(10,10), layout='constrained')

        for cluster_id, ax in zip([1,2,3,4], axes.flatten()):
            ax.plot(years, year_counts_for_plotting_reversed[cluster_id]) #, label=f"Cluster nr {cluster_id}")

#             from sklearn.linear_model import LinearRegression
#             reg = LinearRegression().fit(years.reshape(-1, 1), year_counts_for_plotting_reversed[cluster_id])
#             Y = reg.predict(years.reshape(-1, 1))
#             ax.plot(years, Y, label=f"Linear Regression \n coef={round(reg.coef_[0], 4)}", c="red")

            import statsmodels.api as sm
            years_with_constant = sm.add_constant(years)
            model = sm.OLS(year_counts_for_plotting_reversed[cluster_id], years_with_constant).fit()
            Y = model.predict(sm.add_constant(years.reshape(-1, 1)))
            
            ax.plot(years, Y, label=f"Linear Regression \n coef={round(model.params[1], 4)} \n pval={round(model.pvalues[1], 3)}", c="red")

            # # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('nr of heatwaves')
            ax.set_title(f"Cluster nr {cluster_id}")
            ax.set_xticks(np.arange(1940,2024,10))
            ax.legend(loc='upper right') #, ncols=1)
            ax.set_ylim(0,15)

        fig.suptitle(f'Heatwave types over the years \n {self.t2m_name} - {self.cluster_model_name} {self.cluster_type}', fontsize=18)
        if savefig:
            fig.savefig(f"{outpathfigure}/Clusters_over_time_{self.t2m_name}_{self.cluster_model_name}_{self.cluster_type}.png",)
        
        plt.show()
        

    def plot_heatwave_sample(self, x, norm_method, title):
        """Plots one sample (either input or predicted) """
        
        import warnings
        warnings.filterwarnings('ignore')
        
        from HeaT.reconstruction import LONS, LATS
        LONS, LATS = np.meshgrid(LONS, LATS, indexing="ij")


        fig_width = 25
        fig_height = 5
        fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height))

        if norm_method == "standardized":
            vmin, vmax = -3, 3
        elif norm_method == "standardization_cut":
            vmin, vmax = -1, 1
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
            ax.set_title("Day {}".format(t+1))
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
        w_space = 0.03 # 0.03
        fig.subplots_adjust(wspace=w_space, hspace=h_space)
        fig.subplots_adjust(right=0.95)
        ax_cbar = fig.add_axes([0.96, 0.165, 0.01, 0.675])
        fig.colorbar(cs, cax=ax_cbar)

        plt.suptitle(title, fontsize=18)

        plt.show()
        plt.close()
        
    def plot_central_heatwaves(self, to_plot:str, n_closest=1, savefig=False):
        """plot central heatwaves of clusters"""
        assert to_plot in ["std", "raw", "raw_std"], "data not available"
        
        ## first get heatwave centers - how to do that for kmeans, then just grab centroids instead of means 
        ### get the centers of the clusters 
        if self.cluster_model_name == "GMM":
            centers = self.cluster_model.means_ 
        elif self.cluster_model_name == "Kmeanseucl":
            centers = self.cluster_model.cluster_centers_ 

        #find the distance between the heatwave_means and the centers, and select the closest n
        central_heatwave_index = {}
        for i, center in enumerate(centers):
            similarity = cosine_similarity(center.reshape(1, -1), self.heatwave_means)[0] #calculate cosine similarity
            closest_heatwave_i = sorted(range(len(similarity)), key=lambda k: similarity[k])[:n_closest] #get n closest 
            central_heatwave_index[i+1]=closest_heatwave_i
        
        """here I should define which tf_record data I want to plot, now I only do std, but I also want to be 
        able to plot raw values, and both together"""
        
        
        ### Second get the tfrecord data or some other data
        central_heatwave_data = {}        
        for cluster_id, heatwave_indices in central_heatwave_index.items():
            heatwave_mean_list = []
            print(cluster_id, heatwave_indices)
            for i, element in enumerate(self.tf_records.as_numpy_iterator()): #to convert to numpy elements
                if type(heatwave_indices) == int:
                    heatwave_indices = [heatwave_indices] #in case of n_closest==1
                if i in heatwave_indices:
                    heatwave_mean_list.append(element[0])
            heatwave_mean = np.nanmean(np.array(heatwave_mean_list), axis=0)
            central_heatwave_data[cluster_id]=heatwave_mean
                
        #actual plotting...
        for clusterid, heatwave_mean in central_heatwave_data.items():
            self.plot_heatwave_sample(heatwave_mean, 
                             "standardization_cut", 
                             f"Central Heatwave cluster ID {clusterid} \n {self.cluster_model_name} - {self.cluster_type} - n_closest={n_closest}")

    
        