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


from HeaT.reconstruction import plot_sample
from HeaT.TFrecord_utils_ERA_v2 import load_tfrecords_ERA5, parse_function_full_era5


#from scipy stats
def false_discovery_control(ps, *, axis=0, method='bh'):
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps : 1D array_like
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis : int
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method : {'bh', 'by'}
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg [1]_ (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
        [2]_ (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adusted : array_like
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    combine_pvalues
    statsmodels.stats.multitest.multipletests

    Notes
    -----
    In multiple hypothesis testing, false discovery control procedures tend to
    offer higher power than familywise error rate control procedures (e.g.
    Bonferroni correction [1]_).

    If the p-values correspond with independent tests (or tests with
    "positive regression dependencies" [2]_), rejecting null hypotheses
    corresponding with Benjamini-Hochberg-adjusted p-values below :math:`q`
    controls the false discovery rate at a level less than or equal to
    :math:`q m_0 / m`, where :math:`m_0` is the number of true null hypotheses
    and :math:`m` is the total number of null hypotheses tested. The same is
    true even for dependent tests when the p-values are adjusted accorded to
    the more conservative Benjaminini-Yekutieli procedure.

    The adjusted p-values produced by this function are comparable to those
    produced by the R function ``p.adjust`` and the statsmodels function
    `statsmodels.stats.multitest.multipletests`. Please consider the latter
    for more advanced methods of multiple comparison correction.

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    .. [2] Benjamini, Yoav, and Daniel Yekutieli. "The control of the false
           discovery rate in multiple testing under dependency." Annals of
           statistics (2001): 1165-1188.

    .. [3] TileStats. FDR - Benjamini-Hochberg explained - Youtube.
           https://www.youtube.com/watch?v=rZKa4tW2NKs.

    .. [4] Neuhaus, Karl-Ludwig, et al. "Improved thrombolysis in acute
           myocardial infarction with front-loaded administration of alteplase:
           results of the rt-PA-APSAC patency study (TAPS)." Journal of the
           American College of Cardiology 19.5 (1992): 885-891.

    Examples
    --------
    We follow the example from [1]_.

        Thrombolysis with recombinant tissue-type plasminogen activator (rt-PA)
        and anisoylated plasminogen streptokinase activator (APSAC) in
        myocardial infarction has been proved to reduce mortality. [4]_
        investigated the effects of a new front-loaded administration of rt-PA
        versus those obtained with a standard regimen of APSAC, in a randomized
        multicentre trial in 421 patients with acute myocardial infarction.

    There were four families of hypotheses tested in the study, the last of
    which was "cardiac and other events after the start of thrombolitic
    treatment". FDR control may be desired in this family of hypotheses
    because it would not be appropriate to conclude that the front-loaded
    treatment is better if it is merely equivalent to the previous treatment.

    The p-values corresponding with the 15 hypotheses in this family were

    >>> ps = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    ...       0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]

    If the chosen significance level is 0.05, we may be tempted to reject the
    null hypotheses for the tests corresponding with the first nine p-values,
    as the first nine p-values fall below the chosen significance level.
    However, this would ignore the problem of "multiplicity": if we fail to
    correct for the fact that multiple comparisons are being performed, we
    are more likely to incorrectly reject true null hypotheses.

    One approach to the multiplicity problem is to control the family-wise
    error rate (FWER), that is, the rate at which the null hypothesis is
    rejected when it is actually true. A common procedure of this kind is the
    Bonferroni correction [1]_.  We begin by multiplying the p-values by the
    number of hypotheses tested.

    >>> import numpy as np
    >>> np.array(ps) * len(ps)
    array([1.5000e-03, 6.0000e-03, 2.8500e-02, 1.4250e-01, 3.0150e-01,
           4.1700e-01, 4.4700e-01, 5.1600e-01, 6.8850e-01, 4.8600e+00,
           6.3930e+00, 8.5785e+00, 9.7920e+00, 1.1385e+01, 1.5000e+01])

    To control the FWER at 5%, we reject only the hypotheses corresponding
    with adjusted p-values less than 0.05. In this case, only the hypotheses
    corresponding with the first three p-values can be rejected. According to
    [1]_, these three hypotheses concerned "allergic reaction" and "two
    different aspects of bleeding."

    An alternative approach is to control the false discovery rate: the
    expected fraction of rejected null hypotheses that are actually true. The
    advantage of this approach is that it typically affords greater power: an
    increased rate of rejecting the null hypothesis when it is indeed false. To
    control the false discovery rate at 5%, we apply the Benjamini-Hochberg
    p-value adjustment.

    >>> from scipy import stats
    >>> stats.false_discovery_control(ps)
    array([0.0015    , 0.003     , 0.0095    , 0.035625  , 0.0603    ,
           0.06385714, 0.06385714, 0.0645    , 0.0765    , 0.486     ,
           0.58118182, 0.714875  , 0.75323077, 0.81321429, 1.        ])

    Now, the first *four* adjusted p-values fall below 0.05, so we would reject
    the null hypotheses corresponding with these *four* p-values. Rejection
    of the fourth null hypothesis was particularly important to the original
    study as it led to the conclusion that the new treatment had a
    "substantially lower in-hospital mortality rate."

    """
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)


class LatentSpaceAnalysis:
    def __init__(self, clustered,
                year_splits=[[1940,1970],[1990,2020]],
                top_N_nodes=10):
        
        #clustered information and encoded means
        self.clustered = clustered
        self.heatwave_means = clustered.heatwave_means
        self.heatwave_clusters = clustered.y_pred
        self.heatwave_dates = clustered.heatwave_dates
        #load tf_records 
        self.load_tf_records()
        
        #other parameters
        self.early = year_splits[0]
        self.late  = year_splits[1]
        self.N_nodes = top_N_nodes
                
        #call self to split heatwave means depending on year
        self.split_heatwave_means_year()
        
        #call self to split heatwave means depending on cluster
        self.split_heatwave_means_cluster()
        
        #function to calculate difference of cluster means given mean OR indidivudal heatwaves
        self.loaded_sig_model = None
        
        
    # data loaders and organizers
        
    def tf_records_to_list(self):
        """Fuction to convert tf_records into list with arrays"""
        tf_records_list = []

        sample = np.zeros((1, 192, 64, 5, 2))
        prev_date = ""

        for element in self.data_parsed.as_numpy_iterator(): #to convert to numpy elements
            sample[0] = element[0]
            #remove duplicates
            date = str(element[1]) + "_" + str(element[2]) + "_" + str(element[3]) 
            if date == prev_date:
                print("duplicate")
                continue
            else:
#                 tf_records_list.append(sample) #als je dit doet krijg je dezelfde samples erin??
                element_reshaped = element[0].reshape(1, 192, 64, 5, 2)
                tf_records_list.append(element_reshaped)
            prev_date = date

        self.tf_records_list = tf_records_list

    def load_tf_records(self, t2m_name="t2m_minus_thermo", data_path="/home/thappe/data/"):
        """Loading data from tf_records"""
        
        print(f"loading tf records of {t2m_name} at {data_path}")
        
        assert t2m_name in ["t2m", "t2m_dynamic", "t2m_minus_thermo"], "data not available for this var"
        tf_record_file = f"{data_path}/tf_records/TF_record_ERA5_{t2m_name}_1940-2023_standardization_cut.tfrecord"
                
        data = load_tfrecords_ERA5(tf_record_file)
        
        self.data_parsed = data.map(parse_function_full_era5)
        #convert to list
        self.tf_records_to_list()
        
        
    def split_heatwave_means_year(self):
        """From heatwave_means list, split the data into early and late periods"""
        
        heatwaves_early = []
        heatwaves_late = []

        for i in range(self.heatwave_means.shape[0]):
            year = self.heatwave_dates[i].year
            mean = self.heatwave_means.values[i, :]

            if year >= self.early[0] and year <= self.early[1] : #early cat
                heatwaves_early.append(mean)
            elif year >= self.late[0] and year <= self.late[1]: #late cat
                heatwaves_late.append(mean)
                
        self.heatwave_means_early = heatwaves_early
        self.heatwave_means_late = heatwaves_late
        
        self.heatwaves_early_mean = np.mean(np.array(self.heatwave_means_early), axis=0)
        self.heatwaves_late_mean = np.mean(np.array(self.heatwave_means_late), axis=0)
        
        
    def split_heatwave_means_cluster(self):
        """From heatwave_means list, split the data into the k-cluster assignments"""

        
        self.heatwave_means_per_cluster = {}
        self.heatwave_mean_of_cluster = {}
        self.heatwaves_mean_closest_heatwaves_cluster = {}


        for k in range(self.clustered.cluster_k):
            clusterID = k+1
            indices = np.where(self.heatwave_clusters==k)[0]
            heatwave_means_of_this_cluster = self.heatwave_means.to_numpy()[indices]
            self.heatwave_means_per_cluster[clusterID]=(heatwave_means_of_this_cluster)

            mean_of_cluster = np.mean(heatwave_means_of_this_cluster, axis=0)
            self.heatwave_mean_of_cluster[clusterID]=mean_of_cluster
            
            closest_heatwaves = self.clustered.central_heatwaves_indices[clusterID]
            closest_heatwave_means_of_this_cluster = self.heatwave_means.to_numpy()[closest_heatwaves]
            closest_mean = np.mean(closest_heatwave_means_of_this_cluster, axis=0)
            self.heatwaves_mean_closest_heatwaves_cluster[clusterID]=closest_mean

            
        ## calculate differences
        ks = list(np.arange(1,self.clustered.cluster_k+1,1))
        self.heatwave_difference_mean_of_cluster = {}
        self.heatwave_difference_mean_of_remainingclusters = {}
        self.heatwave_difference_top_nodes = {}
        self.heatwave_difference_mean_of_remainingclusters_split = {}
        self.heatwave_difference_top_nodes_split = {}
        
        for i, cluster_id in enumerate(ks):
            remaining = ks[:i] + ks[i+1:]
            values = self.heatwave_mean_of_cluster[cluster_id]

            #Take the indidivudal difference with other clusters
            differences_split = {}
            top_indices_split = {}
            for remaining_id in remaining:
                diff = abs(values - self.heatwave_mean_of_cluster[remaining_id])
                differences_split[remaining_id]=diff
                top_indices = np.argsort(diff)[-self.N_nodes:][::-1]
                top_indices_split[remaining_id]=top_indices
                
            self.heatwave_difference_top_nodes_split[cluster_id] = top_indices_split    
            self.heatwave_difference_mean_of_remainingclusters_split[cluster_id]=differences_split
            
            #also take the combined difference with respect to mean of other clusters 
            others = []
            for remaining_id in remaining:
                others.append(self.heatwave_mean_of_cluster[remaining_id])
            others_mean = np.mean(np.array(others), axis=0)
            self.heatwave_difference_mean_of_remainingclusters[cluster_id]=others_mean
            
            difference = abs(values - others_mean)
            self.heatwave_difference_mean_of_cluster[cluster_id]=difference
            
            top_indices = np.argsort(difference)[-self.N_nodes:][::-1]
            self.heatwave_difference_top_nodes[cluster_id] = top_indices
            
    # VAE model loading reconstructing
    
    def reconstruct_from_saved_model(self, sample, 
                            data_path='/home/thappe/data/VAE_MODEL/entire_model_VAE',
                            to_plot=True):
                
        ###
        """
        To take 1 sample, load the VAE model and use this to reconstruct the sample.
        """

        if self.loaded_sig_model == None:
            self.loaded_sig_model = tf.saved_model.load(
                f"{data_path}/VAE3D_3D_noRSDS_5d_TRANSFER_L128_B0.01_F4_AUGMENTED_final_with_signatures")

        if to_plot:
            x_t = np.transpose(sample[0], (1,0,2,3))
            warnings.filterwarnings("ignore")
            plot_sample(x_t, "standardization_cut", "True Sample")

        #encode, reparameterize, and decode
        sample = sample.astype(np.float32)

        out = self.loaded_sig_model.signatures["encode"](x=sample)
        z = self.loaded_sig_model.signatures["reparameterize"](mean=out["mean"], logvar=out["logvar"])
        predicted = self.loaded_sig_model.signatures["decode"](z=z["z"], apply_sigmoid=tf.convert_to_tensor(False))
        predicted = predicted['decoded'][0,:,:,:,:]


        if to_plot:
            #plot reconstructed
            predicted_transposed = np.transpose(predicted, (1, 0, 2, 3))
            plot_sample(predicted_transposed, "standardization_cut", "Predicted Sample")

        #moet ik dit nog aan self toekennen?
        return predicted, predicted_transposed
    
    
    def change_Z_tensor(self, x, N, V):
    
        assert len(N)==len(V), "length of nodes to change and values are not equal"

        # Wrap axis 0 index for each (since batch dimension = 0)
        indices = tf.constant([[0, p] for p in N])
        values = tf.constant(V, dtype=tf.float32)
        # Update tensor
        x_updated = tf.tensor_scatter_nd_update(x, indices, values)

        return x_updated



    def reconstruct_sample_changed_Z(self, sample, N_nodes_to_change=[1,50,100], New_values=[-4,-4,-4],
                                data_path='/home/thappe/data/VAE_MODEL/entire_model_VAE',
                                to_plot=True):

        ###
        """
        To take 1 sample, load the VAE model and use this to reconstruct the sample.
        """
        if self.loaded_sig_model == None:
            self.loaded_sig_model = tf.saved_model.load(
                f"{data_path}/VAE3D_3D_noRSDS_5d_TRANSFER_L128_B0.01_F4_AUGMENTED_final_with_signatures")

        x_t = np.transpose(sample[0], (1,0,2,3))
        if to_plot:
            warnings.filterwarnings("ignore")
            plot_sample(x_t, "standardization_cut", "True Sample")

        #encode, reparameterize, and decode
        sample = sample.astype(np.float32)

        out = self.loaded_sig_model.signatures["encode"](x=sample)
        z = self.loaded_sig_model.signatures["reparameterize"](mean=out["mean"], logvar=out["logvar"])

        ###CHANGE VALUES 
        x_updated = self.change_Z_tensor(z["z"], N_nodes_to_change, New_values) 

        #decode with new values
        predicted = self.loaded_sig_model.signatures["decode"](z=x_updated, apply_sigmoid=tf.convert_to_tensor(False))
        predicted = predicted['decoded'][0,:,:,:,:]

#         print(f"Z values changed of Nodes {N_nodes_to_change} to {New_values}")

        
        predicted_transposed = np.transpose(predicted, (1, 0, 2, 3))
        diff = predicted_transposed - x_t
        
        if to_plot:
            #plot reconstructed
            plot_sample(predicted_transposed, "standardization_cut", "Changed Sample")
            plot_sample(diff, "standardization_cut", "Difference in samples (changed - orginal)")


        return predicted, predicted_transposed, diff, z["z"], x_updated
    
    def change_in_cluster_pred(self, z_old, z_new, to_plot=True):
        """Given z old and z new (heatwave means) of one sample, how did the cluster assignment changed?"""

        new_cluster = self.clustered.cluster_model.predict(z_new)[0] + 1 
        old_cluster = self.clustered.cluster_model.predict(z_old)[0] + 1

        if to_plot:
            plt.plot(np.arange(128), z_new.numpy()[0], label="New")
            plt.plot(np.arange(128), z_old.numpy()[0], label="Old")
            plt.legend()
            plt.title(f"Cluster assignment changed from {old_cluster} to {new_cluster}")
            plt.show()

        return new_cluster, old_cluster
    
    def wrapper_off(self, sample_ids=None, to_plot=True, SPLIT_CLUSTERS=False):
        """I need a wrapper to do heatwaves, change Z, plot and predict
        
        for a given cluster, and given heatwaves do;
        - select top nodes, change values to mean e.g. turning some nodes "off"
        """
        
        print(f"SPLIT_CLUSTERS is {SPLIT_CLUSTERS}. If True it will calculate the difference per cluster independently, else it will calculate the difference for cluster A with BCD combined.")
        
        #
        dict_of_interest = {}
        
        if sample_ids is not None:
            print("sample ids are not central heatwaves indices")
            dict_of_interest[9999] = sample_ids
            
        elif sample_ids is None:
            dict_of_interest = self.clustered.central_heatwaves_indices
        
        if not SPLIT_CLUSTERS:
            """Combine the values of all other clusters to calculate differences"""

            wrapper_dict = {}
            for cluster_id, sample_ids in dict_of_interest.items():
                # save the P_changed_t so I can plot the mean changes from the cluster, 
                print(cluster_id, sample_ids)
                cluster_dict = {}
                for sample in sample_ids:
                    #select cluster_id for this sample
                    cluster_id_ = self.heatwave_clusters[sample] + 1   

                    #chceck which nodes to change and to which values
                    Nodes_to_change =  self.heatwave_difference_top_nodes[cluster_id_]
                    New_values = self.heatwave_difference_mean_of_remainingclusters[cluster_id_][self.heatwave_difference_top_nodes[cluster_id_]]

                    # Change sample and plot changed reconstructed
                    p_changed, p_changed_t, diff, z_old, z_new = self.reconstruct_sample_changed_Z(
                        sample=self.tf_records_list[sample], 
                        N_nodes_to_change=Nodes_to_change,
                        New_values = New_values,
                        to_plot=to_plot)

                    # has the prediction changed?
                    new_pred, old_pred = self.change_in_cluster_pred(z_old, z_new, to_plot=to_plot)

                    # save info
                    cluster_dict[sample]=[cluster_id_, p_changed_t, diff, new_pred, old_pred]

                wrapper_dict[cluster_id]=cluster_dict
            self.wrapper_off_dict = wrapper_dict
            
            
        elif SPLIT_CLUSTERS:
            wrapper_dict = {}
            for cluster_id, sample_ids in dict_of_interest.items():
                # save the P_changed_t so I can plot the mean changes from the cluster, 
                print(cluster_id, sample_ids)
                cluster_dict = {}
            
                # now iterate over remaining clusters 
                remaining = [key for key in np.arange(1, self.clustered.cluster_k+1) if key != cluster_id] #to get the remaining ids
                for remaining_id in remaining: 
                
                    remaining_dict = {}                    
                    for sample in sample_ids:
                        #select cluster_id for this sample
                        cluster_id_ = self.heatwave_clusters[sample] + 1   
                        
                        """The nodes and values depend on the cluster of the sample"""
                        #mean values of remaining cluster
                        mean_values_of_cluster = self.heatwave_difference_mean_of_cluster[remaining_id] 
                        top_nodes_dict = self.heatwave_difference_top_nodes_split[cluster_id] #top nodes per cluster
                        
                        Nodes_to_change = top_nodes_dict[remaining_id] 
                        New_values = mean_values_of_cluster[Nodes_to_change] #values of other cluster mean

                        # Change sample and plot changed reconstructed
                        p_changed, p_changed_t, diff, z_old, z_new = self.reconstruct_sample_changed_Z(
                            sample=self.tf_records_list[sample], 
                            N_nodes_to_change=Nodes_to_change,
                            New_values = New_values,
                            to_plot=to_plot)
                        

                        # has the prediction changed?
                        new_pred, old_pred = self.change_in_cluster_pred(z_old, z_new, to_plot=to_plot)

                        # save info
                        remaining_dict[sample]=[p_changed_t, diff, new_pred, old_pred]
                    cluster_dict[remaining_id]=remaining_dict
                wrapper_dict[cluster_id]=cluster_dict
            self.wrapper_off_dict = wrapper_dict
            
      
        
    def wrapper_on(self, closest_mean=True, to_plot=True):
        """I need a wrapper to do heatwaves, change Z, plot and predict
        
        for a given cluster A , do;
        - for each other cluster B; select top nodes, change values to mean of nodes 
        of given cluster A, to turn them "on" 
        
        """
        
        wrapper_dict_on = {}
        
        for cluster_id, top_nodes_dict in self.heatwave_difference_top_nodes_split.items():
            mean_values_of_cluster = self.heatwave_difference_mean_of_cluster[cluster_id] #the values of this cluster to change to
            if closest_mean:
                #take the values from the n-closest heatwaves
                mean_values_of_cluster = self.heatwaves_mean_closest_heatwaves_cluster[cluster_id]
            
            # now iterate over remaining clusters 
            remaining = [key for key in self.heatwave_difference_top_nodes if key != cluster_id] #to get the remaining ids
            dict_ID = {}
            print(f"Cluster {cluster_id}")
#             print("top nodes dict", top_nodes_dict)
            for remaining_cluster_id in remaining:
                #top nodes and thus values depend on remaining cluster
                top_nodes = top_nodes_dict[remaining_cluster_id]
#                 print("top nodes are:", top_nodes)
                New_values = mean_values_of_cluster[top_nodes]                
                # select heatwaves
                heatwave_ids = np.argwhere(self.heatwave_clusters == (remaining_cluster_id-1))[:,0]
                
                #to save info
                total_count = heatwave_ids.shape[0]
                count_changed = 0
                diff_not_changed = []
                diff_changed = []
                new_reconstructed = []
                for sample in heatwave_ids:
                    # Change sample and plot changed reconstructed
                    p_changed, p_changed_t, diff, z_old, z_new = self.reconstruct_sample_changed_Z(sample=self.tf_records_list[sample],
                                                                                             N_nodes_to_change=top_nodes,
                                                                                             New_values = New_values,
                                                                                             to_plot=to_plot)
                    # has the prediction changed?
                    new_pred, old_pred = self.change_in_cluster_pred(z_old, z_new, to_plot=to_plot)
                    if new_pred != old_pred:
                        #print("changed")
                        count_changed += 1 
                        diff_changed.append(diff)
                        new_reconstructed.append(p_changed_t)
                    else:
                        diff_not_changed.append(diff)
                        
                #get info 
                diff_not_changed_mean = np.nanmean(np.array(diff_not_changed), axis=0)
                diff_changed_mean = np.nanmean(np.array(diff_changed), axis=0)
                new_reconstructed_mean = np.nanmean(np.array(new_reconstructed), axis=0)
                percentage_changed = (count_changed/total_count) * 100
#                 print(remaining_cluster_id, count_changed, total_count)
                # save info for this cluster
                dict_ID[remaining_cluster_id]={"total heatwaves":total_count, 
                                               "count changed":count_changed, 
                                               "percentage changed":percentage_changed,
                                               "changed reconstruction not-changed heatwaves":diff_not_changed_mean,
                                               "changed reconstruction changed heatwaves":diff_changed_mean,
                                               "new reconstructed heatwave mean": new_reconstructed_mean}
                
            wrapper_dict_on[cluster_id]=dict_ID
        self.wrapper_on_dict = wrapper_dict_on
    
            

    #plotting functions
    
    def plot_period_means(self):
        """Plot the difference in heatwave means of the different periods"""

        
        x=np.arange(128)
        plt.plot(x, self.heatwaves_early_mean, label=f"{self.early[0]}-{self.early[1]}", c="b")
        plt.plot(x, self.heatwaves_late_mean, label=f"{self.late[0]}-{self.late[1]}", c="r")

        plt.xlabel("Latent Dimensions")
        plt.ylabel("Mean value per cluster")
        plt.legend()
        plt.xticks(ticks=np.arange(0, 128, 5))  # Show every 5th label to avoid clutter

        plt.tight_layout()
        plt.rcParams["figure.figsize"] = (20, 6)  # width=30 inches, height=6
        plt.show()
        
    def plot_cluster_means(self, DIFF=False):
        """Plot the difference in heatwave means of the different clusters, 
        or the difference between them"""

        
        dict_for_plot = self.heatwave_mean_of_cluster
        ylabel = "Mean value"
        title = "Mean value of heatwaves"
        
        if DIFF:
            dict_for_plot = self.heatwave_difference_mean_of_cluster
            ylabel = "Difference with others"
            title = "Difference between clusters"

        
        for clusterID, y in dict_for_plot.items():
            x=np.arange(128)
            plt.plot(x, y, label=f"Cluster {clusterID}")

        plt.xlabel("Latent Dimensions")
        plt.ylabel(f"{ylabel} per cluster")
        plt.title(title)

        plt.legend()
        plt.xticks(ticks=np.arange(0, 128, 5))  # Show every 5th label to avoid clutter
        plt.tight_layout()


        plt.rcParams["figure.figsize"] = (20, 6)  # width=30 inches, height=6
        plt.show()
        
    def plot_cluster_important_nodes(self):
        
        for cluster_id, difference in self.heatwave_difference_mean_of_cluster.items():
            top_indices = self.heatwave_difference_top_nodes[cluster_id]
            
            #plot
            x=np.arange(128)
            plt.plot(x, difference, label=f"Cluster {cluster_id}")


            plt.xlabel("Latent Dimensions")
            plt.ylabel("Difference between mean value per cluster and others")
            plt.legend()
            plt.tight_layout()

            # Default ticks every 5
            ticks = list(np.arange(0, 128, 5))

            # Ensure top_indices are included in ticks
            for idx in top_indices:
                if idx not in ticks:
                    ticks.append(idx)
            ticks = sorted(ticks)

            plt.xticks(ticks=ticks)
            plt.tight_layout()

            # Highlight top indices
            ax = plt.gca()
            for tick_label in ax.get_xticklabels():
                tick_label.set_rotation(90)  # rotate labels 45 degrees

                value = int(tick_label.get_text())
                if value in top_indices:
                    tick_label.set_color('red')
                    tick_label.set_fontsize(10)
                else:
                    tick_label.set_color('black')
                    tick_label.set_fontsize(10)

            plt.title(f"Cluster {cluster_id}, top {self.N_nodes} important nodes (highest difference)")
            plt.rcParams["figure.figsize"] = (20, 6)  # width=30 inches, height=6
            plt.show()
            
    def trend_of_nodes(self, to_plot=True, pvalthresh=0.05, fdc=True):
        import statsmodels.api as sm

        heatwave_years = []
        for date in self.heatwave_dates:
            heatwave_years.append(date.year)
        heatwave_years = np.array(heatwave_years)

        self.trend_per_node = []
        self.pvals_per_node = []
        for nodeN in np.arange(128):
            nodeinfo = self.heatwave_means.to_numpy()[:, nodeN]
            #fit lin regression
            years_with_constant = sm.add_constant(heatwave_years)
            model = sm.OLS(nodeinfo, years_with_constant).fit()
            Y = model.predict(sm.add_constant(heatwave_years.reshape(-1, 1)))
            self.trend_per_node.append(model.params[1])
            self.pvals_per_node.append(model.pvalues[1])

        if fdc == True:
            pvals_adjusted = false_discovery_control(self.pvals_per_node)
#             print(f"pvalues changed from {self.pvals_per_node} to {pvals_adjusted}")
            self.pvals_per_node = pvals_adjusted
            
        pvals_per_node_sign = [i for i, p in enumerate(self.pvals_per_node) if p <= pvalthresh]

        if to_plot:
            for i, p in enumerate(self.pvals_per_node):
                if p <= pvalthresh:
                    plt.scatter(i, self.trend_per_node[i], c='r')
                else:
                    plt.scatter(i, self.trend_per_node[i], c='b')

            plt.ylim(-0.01, 0.01)
            plt.title(f"Lin trend per year per node \n pval={pvalthresh} fdc={fdc}")
            plt.rcParams["figure.figsize"] = (20, 6)  # width=30 inches, height=6
            plt.show()

    def changes_by_trends(self, increment_years=100, to_plot=False, pvalthresh=0.05):
        
        #get important nodes and new values based on trends
        significant_nodes = np.argwhere(np.array(self.pvals_per_node) <= pvalthresh)[:,0]
        print("nr of sign nodes:", significant_nodes.shape[0])
        print(significant_nodes, type(significant_nodes))
        
        increment_to_add_peryear = np.array(self.trend_per_node)[significant_nodes] #based on the trend of those nodes & x_years 
        increment_to_add = increment_to_add_peryear * increment_years

        self.dict_changed_by_trends = {}
        all_heatwave_changes = []

        for k in range(self.clustered.cluster_k):
            clusterID = k+1
            heatwave_ids = np.where(self.heatwave_clusters==k)[0]
            total_count = heatwave_ids.shape[0]
            count_changed = 0
            diff_not_changed = []
            diff_changed = []
            for sample in heatwave_ids:
                #calculate new values based on means +/- increment
                New_values = self.heatwave_means.to_numpy()[sample][significant_nodes] + increment_to_add #means of this sample + increment 

                #change the prediction and reconsturction
                p_changed, p_changed_t, diff, z_old, z_new = self.reconstruct_sample_changed_Z(
                    sample=self.tf_records_list[sample], 
                    N_nodes_to_change=significant_nodes,          
                    New_values = New_values,
                    to_plot=to_plot)
                
                # has the prediction changed?
                new_pred, old_pred = self.change_in_cluster_pred(
                    z_old, z_new, 
                    to_plot=to_plot)

                all_heatwave_changes.append(diff)

                #save info... 
                if new_pred != old_pred:
                    #print("changed")
                    count_changed += 1 
                    diff_changed.append(diff)
                else:
                    diff_not_changed.append(diff)

            #get info 
            diff_not_changed_mean = np.nanmean(np.array(diff_not_changed), axis=0)
            diff_changed_mean = np.nanmean(np.array(diff_changed), axis=0)
            percentage_changed = (count_changed/total_count) * 100

            # save info for this cluster
            self.dict_changed_by_trends[clusterID]={"total heatwaves":total_count, 
                                           "count changed":count_changed, 
                                           "percentage changed":percentage_changed,
                                           "changed reconstruction not-changed heatwaves":diff_not_changed_mean,
                                           "changed reconstruction changed heatwaves":diff_changed_mean,}

        self.all_heatwave_changes_mean = np.nanmean(np.array(all_heatwave_changes), axis=0)
        
        
    def trend_of_nodes_split(self, to_plot=False, pvalthresh=0.05, fdc=True, encircle_top_nodes=False):

        import statsmodels.api as sm

        self.trend_per_node_per_cluster = {}
        self.pvals_per_node_per_cluster = {}

        for k in range(self.clustered.cluster_k):
            heatwave_ids = np.argwhere(self.heatwave_clusters == (k))[:,0]
            print(heatwave_ids.shape)

            heatwave_years = []
            for date in self.heatwave_dates:
                heatwave_years.append(date.year)
            heatwave_years = np.array(heatwave_years)

            heatwave_years = heatwave_years[heatwave_ids]
            heatwave_means = self.heatwave_means.to_numpy()[heatwave_ids]

            trend_per_node = []
            pvals_per_node = []
            for nodeN in np.arange(128):
                nodeinfo = heatwave_means[:, nodeN]
                #fit lin regression
                years_with_constant = sm.add_constant(heatwave_years)
                model = sm.OLS(nodeinfo, years_with_constant).fit()
                Y = model.predict(sm.add_constant(heatwave_years.reshape(-1, 1)))
                trend_per_node.append(model.params[1])
                pvals_per_node.append(model.pvalues[1])

            if fdc == True:
                pvals_per_node = false_discovery_control(pvals_per_node)
    
            pvals_per_node_sign = [i for i, p in enumerate(pvals_per_node) if p <= pvalthresh]
             
            label_plotted=False
            stat_label=False
            if to_plot:
                for i, p in enumerate(pvals_per_node):   
                    
                    if p <= pvalthresh:
                        plt.scatter(i, trend_per_node[i], c="r",
                                   label="Stat. Sign" if not stat_label else None)
                        stat_label =True
                        
                    else:
                         plt.scatter(i, trend_per_node[i], c="b")

                        

                    
                    if encircle_top_nodes==True:
                        if i in self.heatwave_difference_top_nodes[k+1]:
                            plt.scatter(i, trend_per_node[i],
                                        s=200,                # bigger size
                                        facecolors='none',    # hollow circle
                                        edgecolors='green',    # highlight color
                                        linewidths=2,
                                        label="Top Nodes" if not label_plotted else None)
                            label_plotted=True
    

                plt.ylim(-0.02, 0.02)
                plt.legend()
                plt.title(f"Lin trend per year per node for cluster {k+1} \n pval={pvalthresh} fdc={fdc}")
                plt.rcParams["figure.figsize"] = (20, 6)  # width=30 inches, height=6
                plt.show()
           

            self.trend_per_node_per_cluster[k+1]=trend_per_node
            self.pvals_per_node_per_cluster[k+1] = pvals_per_node

    def changes_by_trends_split(self, increment_years=100, to_plot=False, pvalthresh=0.05, TOP_NODES=False):
        
        """
        Depending on the cluster, take the relevant trends and pvals to see which nodes to change
        use increment_years to decide how big the change would be - calculated with the respective trends
        
        if TOP_NODES=True, nodes that are changed are the most important ones of the represpective cluster, 
        instead of the nodes that are stat. sign. changing
        
        per heatwave cluster we save the changed pattern in a dictionary
        """
        

        self.dict_changed_by_trends_split = {}
        self.dict_heatwave_means_with_increments_split = {}
        self.dict_heatwave_means_with_increments_split_concatted = {}
        
        for k in range(self.clustered.cluster_k):
            clusterID = k+1
            heatwave_ids = np.where(self.heatwave_clusters==k)[0]
            total_count = heatwave_ids.shape[0]
            count_changed = 0
            diff_not_changed = []
            diff_changed = []
            
            z_news = []
            new_heatwaves = []
            new_central_heatwaves = []

            
            #increment to add should change depending on cluster, so should the significant nodes 
            trends = self.trend_per_node_per_cluster[clusterID]
            pvals  = self.pvals_per_node_per_cluster[clusterID]
            
            significant_nodes = np.argwhere(np.array(pvals) <= pvalthresh)[:,0]
            
            if TOP_NODES==True:
                print("Top nodes is true, changing those instead of the sign. ones")
                significant_nodes = self.heatwave_difference_top_nodes[k+1]

            
            increment_to_add_peryear = np.array(trends)[significant_nodes] #based on the trend of those nodes & x_years 
            increment_to_add = increment_to_add_peryear * increment_years
            
            
            
            for sample in heatwave_ids:
                #calculate new values based on means +/- increment
                New_values = self.heatwave_means.to_numpy()[sample][significant_nodes] + increment_to_add #means of this sample + increment 

                #change the prediction and reconsturction
                p_changed, p_changed_t, diff, z_old, z_new = self.reconstruct_sample_changed_Z(
                    sample=self.tf_records_list[sample],
                    N_nodes_to_change=significant_nodes,
                    New_values = New_values,
                    to_plot=to_plot)
                
                z_news.append(z_new.numpy())
                self.dict_heatwave_means_with_increments_split_concatted[sample]=z_new.numpy()
                new_heatwaves.append(p_changed_t)
                
                if sample in self.clustered.central_heatwaves_indices[clusterID]:
                    #central heatwave
                    new_central_heatwaves.append(p_changed_t)
                
                # has the prediction changed?
                new_pred, old_pred = self.change_in_cluster_pred(z_old, z_new, to_plot=to_plot)


                #save info... 
                if new_pred != old_pred:
                    #print("changed")
                    count_changed += 1 
                    diff_changed.append(diff)
                else:
                    diff_not_changed.append(diff)

            #get info 
            diff_not_changed_mean = np.nanmean(np.array(diff_not_changed), axis=0)
            diff_changed_mean = np.nanmean(np.array(diff_changed), axis=0)
            percentage_changed = (count_changed/total_count) * 100
            new_heatwaves_mean = np.nanmean(np.array(new_heatwaves), axis=0)
            new_central_heatwaves_mean = np.nanmean(np.array(new_central_heatwaves), axis=0)


            # save info for this cluster
            self.dict_changed_by_trends_split[clusterID]={"total heatwaves":total_count, 
                                           "count changed":count_changed, 
                                           "percentage changed":percentage_changed,
                                           "changed reconstruction not-changed heatwaves":diff_not_changed_mean,
                                           "changed reconstruction changed heatwaves":diff_changed_mean,
                                           "new heatwaves with increment":new_heatwaves_mean,
                                           "new central heatwaves with increment":new_central_heatwaves_mean}
            
            self.dict_heatwave_means_with_increments_split[clusterID]=np.array(z_news)
            
 



        