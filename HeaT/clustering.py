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

from HeaT.TFrecord_utils_ERA_v2 import *


class Clustering:
    def __init__(self, t2m_name, VAE_model, data_path, 
                 standardization_method="standardization_cut",
                 cluster_model_name="GMM", cluster_type="fit", cluster_k=4, random_state=2023):
        
        assert t2m_name in ["t2m", "t2m_dynamic", "t2m_minus_thermo"], "data not available for this var"
        self.t2m_name = t2m_name
        self.t2m_data = None
        
        self.VAE_model = VAE_model
        self.data_path = data_path
        
        self.cluster_model_name = cluster_model_name
        self.cluster_type = cluster_type
        self.cluster_k = cluster_k #default is 4, nr of clusters
        self.random_state = random_state
        
        self.heatwave_means = None
        self.heatwave_dates = None
        
        self.tf_records = None 
        self.stand_method = standardization_method
        
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
                                     skiprows=0, header=None)
        
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
        
        self.cluster_model.random_state = self.random_state #assign random state for consistency
        
                
    def load_tf_records(self):
        """
        To load in tf_records (standardized cut) from given t2m data.
        """
        
        assert self.stand_method in ["standardization_cut", "RAW", "ann_var_removed"], "data not available"
        
        tf_record_file = f"/home/thappe/data/tf_records/TF_record_ERA5_{self.t2m_name}_1940-2023_{self.stand_method}.tfrecord"
        data = load_tfrecords_ERA5(tf_record_file)
        data_parsed = data.map(parse_function_full_era5)
        self.tf_records = data_parsed
        
        
    def load_t2m_data(self):
        name_path_dict = {
            't2m': ('t2m','ERA5/1940-2023_T2M_westEU_June23-JA-Sept7_LENTISGRID.nc','ERA5/T2M_westEU_LENTIS_1940-2023_90thp_15d.nc'),
            't2m_dynamic': ("T2M_dynamic", '/T2M_decomposition/T2M_westEU_June23-JA-Sept7_LENTIS=True_decomposed_1940-2023_window=20_CV=False_alpha=10.nc', '/T2M_decomposition/90th_percentile/T2M_dynamic_westEU_LENTIS_1940-2023.nc'),
            't2m_minus_thermo': ("T2M_minus_thermo",'/T2M_decomposition/T2M_minus_thermo_westEU_LENTIS_1940-2023.nc', '/T2M_decomposition/90th_percentile/T2M_minus_thermo_westEU_LENTIS_1940-2023.nc')}
        self.t2m_data = xr.open_dataset(f"/home/thappe/data/{name_path_dict[self.t2m_name][1]}")[name_path_dict[self.t2m_name][0]].astype("float16")

        if self.t2m_name == 't2m':
            self.t2m_data = self.t2m_data.rename({"lon":"longitude", "lat":"latitude"})
        
        
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
 
            if self.cluster_model_name == "GMM":
                #try regulizer
                self.cluster_model.reg_covar = 1e-4
#                 print("tied")
#                 self.covariance_type = "tied" #try to get softer boundaries
                
            self.y_pred = self.cluster_model.fit_predict(self.heatwave_means)
        
        elif self.cluster_type == "transfer": #concat datasets and then fit 
            # https://stackoverflow.com/questions/29095769/sklearn-gmm-on-large-datasets 
            lentis_heatwave_means = pd.read_csv(f'{self.data_path}/LENTIS_Heatwave_means_VAE_L128_1000epochs.csv', 
                             skiprows=1, header=None)
            to_cluster = pd.concat([self.heatwave_means, lentis_heatwave_means])
            
            if self.cluster_k != 4:
                print(f"changing nr of k for clustering to {self.cluster_k}")
                if self.cluster_model_name == "GMM":
                    self.cluster_model.n_components = self.cluster_k
                elif self.cluster_model_name == "Kmeanseucl":
                    self.cluster_model.n_clusters = self.cluster_k
            
            
            self.y_pred = self.cluster_model.fit_predict(to_cluster) 

        elif self.cluster_type == "project":
            self.y_pred = self.cluster_model.predict(self.heatwave_means)
            
        #if model is GMM, also get the probabilities 
        if self.cluster_model_name == "GMM":
            self.y_pred_proba = self.cluster_model.predict_proba(self.heatwave_means) 
        
        
    def plot_over_time(self, savefig=False, outpathfigure="", GMM_proba=None):
        "clustering over time, with plot"
        
        ### counting the amount of heatwaves per cluster, per year 
        years = np.arange(1940, 2024, 1)
        year_counts_for_plotting_reversed  = {}

        for cluster_id in [1,2,3,4]: #np.arange(1, k+1, 1) ? 
            year_counts_for_plotting_reversed[cluster_id] = np.zeros_like(years)
        
        for date, cluster in zip(self.heatwave_dates, self.y_pred):
            cluster_id = cluster+1
            year_index = np.where(years==date.year)
            year_counts_for_plotting_reversed[cluster_id][year_index] += 1 #add one to this year and cluster count
            
        if self.cluster_model_name == "GMM":
            print("Model is GMM")
            if GMM_proba == None:
                print("No threshold set, all heatwaves are counted")
            else:
                print(f"Threshold is set to {GMM_proba}")
                
                #reset the counts per year to 0
                for cluster_id in [1,2,3,4]: #np.arange(1, k+1, 1) ? 
                    year_counts_for_plotting_reversed[cluster_id] = np.zeros_like(years)
                
                for cluster_id in np.arange(self.cluster_k):
                    indices = np.argwhere(self.y_pred_proba[:,cluster_id]>=GMM_proba)[:,0]
                    print(cluster_id, indices.shape)

                    cluster_id = cluster_id+1
                    #get the dates
                    dates = self.heatwave_dates[indices]
                    for date in dates:
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
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xticks(np.arange(1940,2024,10))
            ax.legend(loc='upper right') #, ncols=1)
            ax.set_ylim(0,15)

        fig.suptitle(f'Heatwave types over the years \n {self.t2m_name}', fontsize=18) # - {self.cluster_model_name} {self.cluster_type} 
        if savefig:
            fig.savefig(f"{outpathfigure}/Clusters_over_time_{self.t2m_name}_{self.cluster_model_name}_{self.cluster_type}.png",)
        
        plt.show()
        
        
        
    def plot_over_time_multiple(self, startyears:list=[1940,1979], savefig=False, outpathfigure=""):
        "clusters over time with multiple startyears, with plot"
        
        import statsmodels.api as sm
        
        
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

            Ys = []
            for startyear in startyears:
                assert startyear>=1940 and startyear<2024, "startyear has to be between 1940-2023"
                
                years_ = np.arange(startyear, 2024, 1)
                counts_ = year_counts_for_plotting_reversed[cluster_id][(startyear-1940):]
                               
                years_with_constant = sm.add_constant(years_)
                model = sm.OLS(counts_, years_with_constant).fit()
                Y = model.predict(sm.add_constant(years_.reshape(-1, 1)))

                ax.plot(years_, Y, label=f"Lin. Regr. {startyear}-2023 \n coef={round(model.params[1], 4)} \n pval={round(model.pvalues[1], 3)}") #, c="red")

            # # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('nr of heatwaves')
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xticks(np.arange(1940,2024,10))
            ax.legend(loc='upper left') #, ncols=1)
            ax.set_ylim(0,15)

        fig.suptitle(f'Heatwave types over the years', fontsize=18) #\n {self.t2m_name} - {self.cluster_model_name} {self.cluster_type}
        if savefig:
            print(f"saved at {outpathfigure}/Clusters_over_time_{self.t2m_name}_{self.cluster_model_name}_{self.cluster_type}.png ")
            fig.savefig(f"{outpathfigure}/Clusters_over_time_{self.t2m_name}_{self.cluster_model_name}_{self.cluster_type}.png",)
        
        plt.show()
        
    def plot_decadal_means(self, year_blocks:list=[(1940, 1970), (1990, 2020)], savefig=False, outpathfigure=""):
        "clusters over time with multiple startyears, with plot"
        
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

            for decadal_block in year_blocks:
                assert decadal_block[0]>=1940 and decadal_block[1]<2024, "year has to be between 1940-2023"
                
#                 print(decadal_block)
                
                years_ = np.arange(decadal_block[0], decadal_block[1]+1, 1)
                counts_ = year_counts_for_plotting_reversed[cluster_id][(decadal_block[0]-1940):(decadal_block[1]+1-2024)]
               
                mean_ = np.full(years_.shape, np.mean(counts_))
#                 print(years_, (decadal_block[0]-1940), (decadal_block[1]+1-2024), mean_)

                ax.plot(years_, mean_, label=f"Mean {decadal_block[0]}-{decadal_block[1]} \n mean={round(mean_[0],2)}") #, c="red")

            # # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('nr of heatwaves')
            ax.set_title(f"Cluster nr {cluster_id}")
            ax.set_xticks(np.arange(1940,2024,10))
            ax.legend(loc='upper right') #, ncols=1)
            ax.set_ylim(0,15)

        fig.suptitle(f'Heatwave types over the years \n {self.t2m_name} - {self.cluster_model_name} {self.cluster_type}', fontsize=18)
        if savefig:
            fig.savefig(f"{outpathfigure}/Clusters_over_time_{self.t2m_name}_{self.cluster_model_name}_{self.cluster_type}_decblocks.png",)
        
        plt.show()
        

    def plot_heatwave_sample(self, x, title, diff_plot=False):
        """Plots one sample (either input or predicted) """
        
        import warnings
        warnings.filterwarnings('ignore')

        
        from HeaT.reconstruction import LONS, LATS
        LONS, LATS = np.meshgrid(LONS, LATS, indexing="ij")


        fig_width = 25
        fig_height = 5
        fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height))
        for ax in axes.flat:
            ax.remove()

        if self.stand_method == "standardization_cut":
            vmin_s, vmax_s = -1, 1
            vmin_p, vmax_p = -1, 1
            label_p, label_s = "std", "std"
            
        elif self.stand_method == "RAW":
            vmin_s, vmax_s = -90, 10
            vmin_p, vmax_p = 990, 1030
            label_p, label_s = "hPa", "m2/s"
            
            if diff_plot == True:
                vmin_s, vmax_s = -50, 50
                vmin_p, vmax_p = -20, 20
            

        elif self.stand_method == "ann_var_removed":
            vmin_s, vmax_s = -50, 50
            vmin_p, vmax_p = -20, 20
            label_p, label_s = "hPa", "m2/s"
            

        for t in range(5):
            stream = x[:, :, t, 0]
            psl    = x[:, :, t, 1] 
            
            if self.stand_method == "RAW" or self.stand_method == "ann_var_removed": 
                stream = stream / 1000000
                psl    = psl / 100 
        

            #plot stream function
            ax = plt.subplot(2, 5, t + 1, projection=ccrs.PlateCarree())
            
            cs_s = ax.pcolormesh(LONS, LATS, stream, transform=ccrs.PlateCarree(),
                               cmap="PiYG_r", vmin=vmin_s, vmax=vmax_s)
            ax.coastlines()
            ax.set_title("Day {}".format(t+1))
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--')

            
            gl.top_labels    = False
            gl.bottom_labels = False
            gl.right_labels  = False
            if t == 0:
                gl.left_labels = True
            else:
                gl.left_labels = False
            
            if t == 4:
                ax_cbar = fig.add_axes([0.96, 0.515, 0.01, 0.3]) #
                fig.colorbar(cs_s, cax=ax_cbar, label=label_s)

            #plot sea level pressure
            ax = plt.subplot(2, 5, t + 6, projection=ccrs.PlateCarree())
            cs_p = ax.pcolormesh(LONS, LATS, psl, transform=ccrs.PlateCarree(),
                               cmap="bwr", vmin=vmin_p, vmax=vmax_p)
            ax.coastlines()
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--')
            gl.xlocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40])
            
            gl.top_labels   = False
            gl.right_labels = False
            if t == 0:
                gl.left_labels = True
            else:
                gl.left_labels = False    
                
            if t == 4:
                ax_cbar = fig.add_axes([0.96, 0.165, 0.01, 0.3]) 
                fig.colorbar(cs_p, cax=ax_cbar, label=label_p)

        h_space = -0.14
        #w_space = h_space * fig_height / fig_width
        w_space = 0.03 # 0.03
        fig.subplots_adjust(wspace=w_space, hspace=h_space)
        fig.subplots_adjust(right=0.95)

        plt.suptitle(title, fontsize=18)

        plt.show()
        plt.close()
        
        
    def plot_temperature(self, x, title, ANOM=True, filename="", savefig=False,
                         composite=True):

        import warnings
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

        warnings.filterwarnings('ignore')  
    

        fig_width = 60
        fig_height = 10
        fig, axes = plt.subplots(1, 5, figsize=(fig_width, fig_height))
        for ax in axes:
            ax.remove()
        
        mean_tas = self.t2m_data.mean(dim=("time"), skipna=False)

        for t in range(5):
            if not ANOM:
                temp = x[t, :, :] 
                vmin,vmax = 5, 45
                cmap = "YlOrRd"
            elif ANOM:
                temp=  x[t, :, :] - mean_tas
                vmin, vmax= -8, 8
                if composite:
                    vmin, vmax= -3, 3
            cmap="coolwarm"
        
            #plot stream function
            ax = plt.subplot(1, 5, t + 1, projection=ccrs.PlateCarree())
            cs = ax.pcolormesh(self.t2m_data.longitude, self.t2m_data.latitude, temp, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.coastlines(linewidth=4)
            ax.set_title(f"Day {t+1}", fontsize=35)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                          linewidth=2, color='gray', alpha=0.1, linestyle='--')
            
            gl.top_labels    = False
            gl.bottom_labels = True
            gl.right_labels  = False
            if t == 0:
                gl.left_labels = True
            else:
                gl.left_labels = False

            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 35}
            gl.ylabel_style = {'size': 35}

  #         plt.colorbar(cs, fraction=0.016, pad=0.02, label="degrees C")
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Suppose ax is your last subplot (where the colorbar goes)
        axins = inset_axes(ax, 
                           width="5%",    # width relative to ax
                           height="100%", # height relative to ax
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0)

        cbar = plt.colorbar(cs, cax=axins)
        # Set label size
        cbar.set_label("degrees C", fontsize=30)

        # Set tick label size
        cbar.ax.tick_params(labelsize=25)

        h_space = -0.14
        #w_space = h_space * fig_height / fig_width
        w_space = 0.1
        fig.subplots_adjust(wspace=w_space, hspace=h_space)
        fig.subplots_adjust(right=0.95)

        plt.suptitle(title, fontsize=45, y=1.1)
        
        if savefig:
            plt.savefig(filename, bbox_inches='tight', dpi=400)
            
        plt.show()
        plt.close()    
        
    def plot_central_heatwaves(self, n_closest=1, savefig=False):
        """plot central heatwaves of clusters"""
#         assert to_plot in ["std", "raw", "raw_std"], "data not available"
        
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
            closest_heatwave_i = sorted(range(len(similarity)), key=lambda k: similarity[k])[-n_closest:] #get n closest 
            central_heatwave_index[i+1]=closest_heatwave_i
            
        self.central_heatwaves_indices = central_heatwave_index
        
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
                             f"Central Heatwave cluster ID {clusterid} \n {self.cluster_model_name} - {self.cluster_type} - n_closest={n_closest}")
    
    def plot_central_heatwaves_temperature(self, n_closest=1, filename="", savefig=False, temp_name="t2m_minus_thermo"):
        """plot central heatwaves of clusters"""

        if self.cluster_model_name == "GMM":
            centers = self.cluster_model.means_ 
        elif self.cluster_model_name == "Kmeanseucl":
            centers = self.cluster_model.cluster_centers_ 
            
        if type(self.t2m_data) == type(None):
            #t2m data not yet loaded
            self.load_t2m_data()

        if self.t2m_name != temp_name:
            """Plotting different temperature then the self.t2m_data"""
            prev = self.t2m_name
            # load new t2m data
            assert temp_name in ["t2m", "t2m_dynamic", "t2m_minus_thermo"]
            print("Warning! self.t2m_data will be updated to new t2m_name data")
            self.t2m_name = temp_name
            self.load_t2m_data()
            # #return name to original one
            self.t2m_name = prev

            

        #find the distance between the heatwave_means and the centers, and select the closest n
        central_heatwave_index = {}
        for i, center in enumerate(centers):
            similarity = cosine_similarity(center.reshape(1, -1), self.heatwave_means)[0] #calculate cosine similarity
            closest_heatwave_i = sorted(range(len(similarity)), key=lambda k: similarity[k])[:n_closest] #get n closest 
            central_heatwave_index[i+1]=closest_heatwave_i
        
        
        ### Second get the tfrecord data or some other data
        central_heatwave_data = {}        
        for cluster_id, heatwave_indices in central_heatwave_index.items():
#             print(cluster_id, heatwave_indices)
            heatwave_mean_list = []
            for heatwave_indice in heatwave_indices:
                startdate = self.heatwave_dates[heatwave_indice]
                end_date = startdate + pd.Timedelta(days=5)
                t2m_data_heatwave = self.t2m_data.sel(time=slice(startdate, end_date)).values
                heatwave_mean_list.append(t2m_data_heatwave)
                
            heatwave_mean = np.nanmean(np.array(heatwave_mean_list), axis=0)
            central_heatwave_data[cluster_id]=heatwave_mean
                
        #actual plotting...
        for clusterid, heatwave_mean in central_heatwave_data.items():
            self.plot_temperature(heatwave_mean, 
                             f"Central Heatwave cluster ID {clusterid} \n {self.cluster_model_name} - {self.cluster_type} - n_closest={n_closest}",
                                 ANOM=True,
                                 filename=f"{filename}_{clusterid}.png", savefig=savefig)
            
    def plot_central_heatwaves_decadal_changes(self, n_closest=1, year_blocks:list=[(1940, 1970), (1990, 2020)], 
                                               savefig=False):
        """plot changes in central heatwaves of clusters"""
        
        ## first get heatwave centers - how to do that for kmeans, then just grab centroids instead of means 
        ### get the centers of the clusters 
        if self.cluster_model_name == "GMM":
            centers = self.cluster_model.means_ 
        elif self.cluster_model_name == "Kmeanseucl":
            centers = self.cluster_model.cluster_centers_ 

                
        #
        assert len(year_blocks) == 2, "can not make changes for more than 2 timeframes"
        
        years = []
        for date in self.heatwave_dates:
            years.append(date.year) 
            
        
        
        decadal_data = []
        for decadal_block in year_blocks:
            indices = [i for i, year in enumerate(years) if decadal_block[0] <= year <= decadal_block[1]]
            means_block = self.heatwave_means.iloc[indices]
            
            #find the distance between the heatwave_means and the centers, and select the closest n
            central_heatwave_index = {}
            for i, center in enumerate(centers):
                similarity = cosine_similarity(center.reshape(1, -1), means_block)[0] #calculate cosine similarity
                closest_heatwave_i = sorted(range(len(similarity)), key=lambda k: similarity[k])[:n_closest] #get n closest 
                central_heatwave_index[i+1]=closest_heatwave_i


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
            decadal_data.append(central_heatwave_data)
        
        heatwave_changes = {}
        for i in range(len(centers)):
            first_block  = decadal_data[0][i+1]
            last_block = decadal_data[1][i+1]
            difference = last_block - first_block
            heatwave_changes[i+1]=difference
            
                
        #actual plotting...
        for clusterid, heatwave_mean in heatwave_changes.items():
            self.plot_heatwave_sample(heatwave_mean, 
                             f"Difference plot \n Central Heatwave cluster ID {clusterid} \n {self.cluster_model_name} - {self.cluster_type} - n_closest={n_closest}",
                                     diff_plot=True)

        from sklearn.metrics.pairwise import pairwise_distances_argmin, cosine_similarity


    def plot_heatwave_sample_contours(self, x, x_raw, title, diff_plot=False,
                                     cluster_id=999, savefig=False, outpath="",
                                     n_closest=1):
            """Plots one sample (either input or predicted) """
            
            import warnings
            warnings.filterwarnings('ignore')
    
            
            from HeaT.reconstruction import LONS, LATS
            LONS, LATS = np.meshgrid(LONS, LATS, indexing="ij")
    
    
            fig_width = 25
            fig_height = 5
        
            fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height))
            for ax in axes.flat:
                ax.remove()
    
            vmin_s, vmax_s = -1, 1
            vmin_p, vmax_p = -1, 1
            label_p, label_s = "std", "std"
            
            vmin_s_raw, vmax_s_raw = -90, 10
            vmin_p_raw, vmax_p_raw = 990, 1030
            label_p_raw, label_s_raw = "hPa", "m2/s"
                
    
            for t in range(5):
                stream = x[:, :, t, 0]
                psl    = x[:, :, t, 1] 
                stream_raw = x_raw[:, :, t, 0] / 1000000
                psl_raw = x_raw[:, :, t, 1] / 100 
                
                # if self.stand_method == "RAW" or self.stand_method == "ann_var_removed": 
                #     stream = stream / 1000000
                #     psl    = psl / 100 
            
    
                #plot stream function
                ax = plt.subplot(2, 5, t + 1, projection=ccrs.PlateCarree())
                
                cs_s = ax.pcolormesh(LONS, LATS, stream, transform=ccrs.PlateCarree(),
                                   cmap="PiYG_r", vmin=vmin_s, vmax=vmax_s)
    
                contours = ax.contour(LONS, LATS, stream_raw, levels=np.linspace(vmin_s_raw, vmax_s_raw, 11),
                          colors='black', linewidths=0.8, transform=ccrs.PlateCarree())
    
                ax.clabel(contours, inline=True, fontsize=8)
    
                
                ax.coastlines()
                ax.set_title("Day {}".format(t+1))
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
                
                gl.top_labels    = False
                gl.bottom_labels = False
                gl.right_labels  = False
                if t == 0:
                    gl.left_labels = True
                else:
                    gl.left_labels = False
                
                if t == 4:
                    ax_cbar = fig.add_axes([0.96, 0.515, 0.01, 0.3]) #
                    fig.colorbar(cs_s, cax=ax_cbar, label=label_s)
    
                #plot sea level pressure
                ax = plt.subplot(2, 5, t + 6, projection=ccrs.PlateCarree())
                cs_p = ax.pcolormesh(LONS, LATS, psl, transform=ccrs.PlateCarree(),
                                   cmap="bwr", vmin=vmin_p, vmax=vmax_p)
                
                contours = ax.contour(LONS, LATS, psl_raw, levels=np.linspace(vmin_p_raw, vmax_p_raw, 11),
                          colors='black', linewidths=0.8, transform=ccrs.PlateCarree())
    
                ax.clabel(contours, inline=True, fontsize=8)
    
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40])
                
                gl.top_labels   = False
                gl.right_labels = False
                if t == 0:
                    gl.left_labels = True
                else:
                    gl.left_labels = False    
                    
                if t == 4:
                    ax_cbar = fig.add_axes([0.96, 0.165, 0.01, 0.3]) 
                    fig.colorbar(cs_p, cax=ax_cbar, label=label_p)
    
            h_space = -0.14
            #w_space = h_space * fig_height / fig_width
            w_space = 0.03 # 0.03
            fig.subplots_adjust(wspace=w_space, hspace=h_space)
            fig.subplots_adjust(right=0.95)
    
            plt.suptitle(title, fontsize=18)
    
            if savefig == True:
                plt.savefig(f"{outpath}/central_heatwaves_n={n_closest}_contours_cluster{cluster_id}.png", dpi=400)
            plt.show()
            plt.close()


    def plot_central_heatwaves_withcontours(self, n_closest=1, savefig=False):
            """plot central heatwaves of clusters"""
    #         assert to_plot in ["std", "raw", "raw_std"], "data not available"
            
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
                closest_heatwave_i = sorted(range(len(similarity)), key=lambda k: similarity[k])[-n_closest:] #get n closest 
                central_heatwave_index[i+1]=closest_heatwave_i
                
            self.central_heatwaves_indices = central_heatwave_index
            
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
    
            ##  now also load RAW
            tf_record_file = f"/home/thappe/data/tf_records/TF_record_ERA5_{self.t2m_name}_1940-2023_RAW.tfrecord"
            data = load_tfrecords_ERA5(tf_record_file)
            data_parsed_raw = data.map(parse_function_full_era5)
    
            central_heatwave_data_raw = {}        
            for cluster_id, heatwave_indices in central_heatwave_index.items():
                heatwave_mean_list = []
                print(cluster_id, heatwave_indices)
                for i, element in enumerate(data_parsed_raw.as_numpy_iterator()): #to convert to numpy elements
                    if type(heatwave_indices) == int:
                        heatwave_indices = [heatwave_indices] #in case of n_closest==1
                    if i in heatwave_indices:
                        heatwave_mean_list.append(element[0])                    
                heatwave_mean = np.nanmean(np.array(heatwave_mean_list), axis=0)
                central_heatwave_data_raw[cluster_id]=heatwave_mean
                    
            #actual plotting...
            for clusterid, heatwave_mean in central_heatwave_data.items():
                self.plot_heatwave_sample_contours(heatwave_mean, central_heatwave_data_raw[cluster_id], 
                                 f"Central Heatwave cluster ID {clusterid} \n {self.cluster_model_name} - {self.cluster_type} - n_closest={n_closest}",
                                             cluster_id=clusterid, savefig=savefig, outpath="/home/thappe/HeaT/Figures/closest_heatwaves", n_closest=n_closest)
        

    
        