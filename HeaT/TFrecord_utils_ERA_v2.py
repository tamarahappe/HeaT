import numpy as np 
from netCDF4 import Dataset
import os
import tensorflow as tf
import argparse
import pandas as pd

#####
# WRITING TF_Records
####

def loading_heatwave_data(filename:str, years_n:int):
	'''To load in the heatwaves of respective ensemble member
	INPUTS:
	path:str = path where data is stored
	
	Takes in ensemble number and path where data is stored, returns lists with
	years_i: list with respective year
	day0_i: list with start of day of heatwave
	years_ind: index based on year and ensemble
	'''
	
	curdir = os.getcwd()
	heatwaves_df = pd.read_csv(fr'{filename}')

	years = heatwaves_df["year"]
	day0 = heatwaves_df["day 0 (index)"]
	dates = heatwaves_df["start date"]

	#retrieve heaywave event information
	years_i, day0_i, dates_i = [], [], []
	for i in range(len(years)):
		if day0[i] == " ": # or years[i] == "2022": #discard noise 
			continue
		years_i.append(float(years[i]))
		day0_i.append(float(day0[i]))
		dates_i.append(dates[i])

	os.chdir(curdir)
	return years_i, day0_i, dates_i

def preprocessing_v3(variables, path, years_n, 
                     startyear:int=2022, endyear:int=2023,
                     method="standardization_cut"):
    """
    Purpose is to load data of each variable, to cut it to the desired extent,
    and to return the normalized/standardized values

    INPUTS
    variables: list of varariable names
    ensemble member: integer of ensemble member number 
    path: string of the path name where the data is stored
    years_n: integer of the number of years in the ensemble
    extent: list of 4 integers with [lon_min,lon_max,lat_min,lat_max]

    OUTPUTS
    outs: list of arrays, with each array containing one variable 

    """
    print("Data is assumed to be from June-September with 122 days per year")
    import xarray as xr
    
    outs = []
    if type(variables) != list:
        raise ValueError("variables is not in list format")
        
    assert method in ['standardization_cut', 'stand_norm_nans', 'stand_norm_01s',
                      "RAW", "ann_var_removed"], "normalization method not available"

    curdir = os.getcwd()
    os.chdir(path)
    print(curdir, os.getcwd())
    print(startyear, endyear)
    ######
    for var in variables:
        print(f"variable is {var}")
        if var == 'stream':
            var = xr.open_dataset(f"STREAM250_era5_NAExt_0.25degr_1940-2023_JJAS_like_LENTIS.nc")
            var_lons = var.lon
            var_lats = var.lat
            var_cut = var.sel(time=var.time.dt.year>=startyear)
            var_cut = var_cut.sel(time=var_cut.time.dt.year<=endyear)["stream"]
            var_array = var_cut.values
            
            #var_mean = Dataset(fr"ML_prep_MSLP_STREAM/STREAM250_NAExt_1940-2011_TRAIN_15drunMEAN.nc")['stream'][:]
            #var_std = Dataset(fr'ML_prep_MSLP_STREAM/STREAM250_NAExt_1940-2011_TRAIN_15drunSTD.nc')['stream'][:]
            var_mean = Dataset(fr"STREAM250_era5_NAExt_0.25degr_1940-2023_JJAS_like_LENTIS_15dMEAN.nc")['stream'][:]
            var_std = Dataset(fr'STREAM250_era5_NAExt_0.25degr_1940-2023_JJAS_like_LENTIS_15dSTD.nc')['stream'][:]
            
        elif var == 'mslp':
            var = xr.open_dataset(f"MSLP_era5_NAExt_0.25degr_1940-2023_JJAS_like_LENTIS.nc")
            var_lons = var.lon
            var_lats = var.lat
            var_cut = var.sel(time=var.time.dt.year>=startyear)
            var_cut = var_cut.sel(time=var_cut.time.dt.year<=endyear)["msl"]
            var_array = var_cut.values
            
#             var_mean = Dataset(fr'ML_prep_MSLP_STREAM/MSLP_NAExt_1940-2011_TRAIN_15drunMEAN.nc')["msl"][:]
#             var_std = Dataset(fr'ML_prep_MSLP_STREAM/MSLP_NAExt_1940-2011_TRAIN_15drunSTD.nc')["msl"][:]            
            var_mean = Dataset(fr'MSLP_era5_NAExt_0.25degr_1940-2023_JJAS_like_LENTIS_15dMEAN.nc')["msl"][:]
            var_std = Dataset(fr'MSLP_era5_NAExt_0.25degr_1940-2023_JJAS_like_LENTIS_15dSTD.nc')["msl"][:]
            
            
        #now standardization
        var_out= np.empty_like(var_array)		
        if method == 'standardization_cut':
            print("standardization method: out=(X-Xmean)/(3 * Xstd)")
            for year in range(years_n): 
                var_year = (var_array[year*122:(year+1)*122,:,:] - var_mean) / (3 * var_std)
                var_year = np.where(var_year < -1, -1, var_year) #remove outliers 
                var_year = np.where(var_year > 1, 1, var_year)
                var_out[year*122:(year+1)*122,:,:]=var_year
            outs.append(var_out)

        elif method == 'stand_norm_nans':
            print("stand + norma +nans method: out=(standardized_value - -3)/(3 - -3)")
            print("above 1 and below 0 are replaced with np.nan - outliers ")
            for year in range(years_n): 
                var_year = (((var_array[year*122:(year+1)*122,:,:] - var_mean) / var_std ) + 3) / 6
                var_year = np.where(var_year < 0, np.NAN, var_year) #remove outliers 
                var_year = np.where(var_year > 1, np.NAN, var_year)
                var_out[year*122:(year+1)*122,:,:]=var_year
            outs.append(var_out)
            
        elif method == "RAW":
            print("method is RAW, no standardization applied")
            outs.append(var_array)
            
        elif method == "ann_var_removed":
            print("No standardization applied, but interannual variability is removed (fieldmean)")
            mean_field = var_cut.mean(dim=["lat", "lon"]) #take fieldmean over lat and lon
#             print(var_cut.shape, mean_field.shape)
            var_out = var_cut - mean_field
            outs.append(var_out.values)

        elif method == 'stand_norm_01s':
            print("stand + norma +nans method: out=(standardized_value - -4)/(4 - -4)")
            print("above 1 and below 0 are replaced with 1 and 0 respecitvely - outliers ")
            for year in range(years_n): 
                var_year = (((var_array[year*122:(year+1)*122,:,:] - var_mean) / var_std ) + 4) / 8
                var_year = np.where(var_year < 0, 0, var_year) #remove outliers 
                var_year = np.where(var_year > 1, 1, var_year)
                var_out[year*122:(year+1)*122,:,:]=var_year
            outs.append(var_out)

    os.chdir(curdir)
    return outs

def data_to_events_v2(data_list:list, variables:list, event_len:int, years_i:list, dates_i:list, day0_i:list, startyear:int, endyear:int):

    #PURPOSE: to create numpy array with [events, feature1, feature2, feature3]
    #INPUTS
    #data_list = list with data_array for each variable in variables 
    #variables = list with variables inside the data_arrat
    #event_len = number of days before and after day 0
    #ensemble_i = heatwave_list with ensemble member index
    #years_i = heatwave_list with years
    #dates_i = heatwave_list with dates
    #day0_i = heatwave_list with starting day
    #years_ind = heatwave_list with index of ensemble_year
    #OUTPUT
    #events = list with for each event the three variable arrays
    #dates_updated = dates_i filtered for the duplicates
    

    print("Variable order is ", variables[:])
    stream_data = data_list[0]
    print("stream_data.shape is", stream_data.shape)
    psl_data = data_list[1]
    print("psl_data.shape is", psl_data.shape)
    events = []
    
    years_unique = np.unique(years_i) #all the unique years in my dataset 
    years_count = endyear-startyear + 1 #number of years 
        
    assert int(stream_data.shape[0]/122) == years_count, "length of data is not equal to amount of years"
    
    prev_event = []    
    duplicate_count = 0
    dates_updated = []
    for i, year in enumerate(years_i):
        #select the correct data from the variables
        j = np.where(years_unique == year)[0][0]
        index_day = int(j * 122 + 30 - event_len + day0_i[i]) #year_intm june july date and event length
        cur_event = [year, day0_i[i]] 
        if prev_event == cur_event:
            duplicate_count = duplicate_count + 1 
            #print(prev_event, cur_event)
            continue
        else:
            prev_event = cur_event #update memory
            #index day in dataset is the year * days_in_year + 23 (june is 30 - 7days for start of event) + start_of_event_day 
            end = index_day + 2*event_len 
            f1 = stream_data[index_day:end,:,:]
            f2 = psl_data[index_day:end,:,:] 
            event = np.array([f1,f2])
            events.append(np.transpose(event)) #transpose such that shape becomes spatial, temporal, features
            dates_updated.append(dates_i[i])
    print("duplicate count =", duplicate_count)
    return events, dates_updated


def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns a int64_list from a bool/enum/int/uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(event,date):
	"""
	Creates a tf.train.Example message ready to be written to a file.
	"""
	# Create a dictionary mapping the feature name to the tf.train.Example-compatible
	# data type.
	feature = {
	  'features': _bytes_feature(tf.io.serialize_tensor(event)),
	  'year': _int64_feature(int(date[:4])),
	  'month': _int64_feature(int(date[5:7])),
	  'day': _int64_feature(int(date[8:])), 
	  # 'date': _bytes_feature(u"{date}".format(date=date).encode('utf-8')), #is er een tf.io optie voor string?
	}

	# Create a Features message using tf.train.Example.
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

def data_to_TFrecord_v2(t2m:str, heatwave_file:str, path_heatwaves:str, data_path:str, variables:list, path_climate:str, 
                     nyears:int, event_len:int=7, method:str="standardization_cut",
                     startyear:int=2022, endyear:int=2023,):
    '''PURPOSE: to read in data from heatwaves and climate data, preform preprocessing steps, and write a
    TF record file

    INPUTS:
    t2m:str = Name of T2M type 
    path_heatwaves:str = path where heatwave data is stored
    variables:list of strings = variable names of climate data
    path_climate: string = of the path name where the climate data is stored
    nyears: integer =  the number of years of dataset
    event_len: int = number of days before and after event (e.g.7 means 7 days before and 7 days after day0)
    method: str = name of method for preprocessing

    OUTPUTS:
    writes TF record file of all heatwave events 
    '''
    #READ IN HEATWAVE DATA
    years_i, day0_i, dates_i = loading_heatwave_data(f"{path_heatwaves}{heatwave_file}", nyears)
    #READ AND PREPROCESS CLIMATE DATA
    list_of_data = preprocessing_v3(variables, path_climate, nyears, method=method,
                                   startyear=startyear, endyear=endyear) 
    #CREATE HEATWAVE EVENTS
    events, dates_i = data_to_events_v2(list_of_data, variables, event_len, years_i, dates_i, day0_i, startyear, endyear)
    
    #events in verschillenjde tf records, 200mb voor 1 shard, 

    #PROCESS EVENTS INTO TF RECORD FILES
    current_shard = 0
    img_in_current_shard = 0
    SAMPLES_PER_SHARD = len(events) #later zien of dit niet te groot is
    writer = tf.io.TFRecordWriter(f"{data_path}/tf_records/TF_record_ERA5_{t2m}_{startyear}-{endyear}_{method}.tfrecord")
    for i, event in enumerate(events):
        if img_in_current_shard == SAMPLES_PER_SHARD:
            writer.close()
            #open new file
            current_shard += 1
            img_in_current_shard = 0
            writer = tf.io.TFRecordWriter(f"{data_path}/tf_records/TF_record_ERA5_{t2m}_{startyear}-{endyear}_{method}.tfrecord")
        #process current sample and write to file
        tf_example = serialize_example(event, dates_i[i])
        #print(tf_example)
        writer.write(tf_example) #Serializetostring()
        img_in_current_shard += 1
    writer.close()


#####
# READING TF_Records
####

# NOTE; the TF_records are written with 14 days, 7days before and 7days after the event. 
# BUT we clustered them on 5 days only, hence the features[7:12]


def load_tfrecords_ERA5(filename):
    data = tf.data.TFRecordDataset([filename])
    return data

# #TO see how many samples there are in the data
# samples_data_train = data_train_raw.reduce(np.int64(0), lambda x, _: x+1).numpy()
# samples_data_val = data_val_raw.reduce(np.int64(0), lambda x, _: x+1).numpy()
# print(samples_data_train, samples_data_val) #14198 1964 --> removed duplicates; 12164 1714 

#Define features saved in tf_record
feature_description_era5 = {
'features': tf.io.FixedLenFeature([], tf.string),
'year': tf.io.FixedLenFeature([], tf.int64),
'month': tf.io.FixedLenFeature([], tf.int64),
'day': tf.io.FixedLenFeature([], tf.int64),
}


#Create functions to read tf_records
def parse_function_era5(example):
    """
    Takes in example event from the raw data (tensorflow dataset), and returns the original values
    
    if test==True, the example is returned with all information attached. Otherwise just the features are returned
    for the use of model training.
    """
    parsed_example = tf.io.parse_single_example(example, feature_description_era5)

    features = parsed_example['features']
    features = tf.io.parse_tensor(features, tf.float32)
    return features[:,:,7:12,:], features[:,:,7:12,:] #to return only 5 days


#Create functions to read tf_records
def parse_function_augment_era5(example):
    """
    Takes in example event from the raw data (tensorflow dataset), and returns the original values
    
    if test==True, the example is returned with all information attached. Otherwise just the features are returned
    for the use of model training.
    """
    parsed_example = tf.io.parse_single_example(example, feature_description_era5)

    features = parsed_example['features']
    features = tf.io.parse_tensor(features, tf.float32)
    return features[:,:,7:12,:]
    

def parse_function_full_era5(example):
    """
    Takes in example event from the raw data (tensorflow dataset), and returns the original values
    
    if test==True, the example is returned with all information attached. Otherwise just the features are returned
    for the use of model training.
    """
    parsed_example = tf.io.parse_single_example(example, feature_description_era5)
    
    features = parsed_example['features']
    features = tf.io.parse_tensor(features, tf.float32)

    year = parsed_example['year']
    month = parsed_example['month']
    day = parsed_example['day']

    return features[:,:,7:12,:], year, month, day



