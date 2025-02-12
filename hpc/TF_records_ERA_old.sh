#!/bin/bash
#SBATCH -o /home/the410/data/ERA5/tf_records/%j.out
#SBATCH --partition=ivm
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=t.happe@vu.nl
#SBATCH -t 8:00:00
echo start of job

module load python3

cd $HOME

#eval "$(conda shell.bash hook)"
#conda activate gdbscan

#SET VARS

PATH_HEATWAVES='/scistor/ivm/the410/data/heatwaves/ERA5/ERA5_WestEU_2000-2021_heatwave_clusters_p90mp21_lentisgrid.csv'
#VARIABLES= ['stream250', 'rsds', 'psl']
PATH_CLIM="/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/"

cd $HOME/data/ERA5/tf_records/

python TFrecord_loop_ERA.py -path_heatwaves $PATH_HEATWAVES -path_climate $PATH_CLIM -years_n 23 -extent_name "NAext2" -event_len 7 -method "standardization_cut"

echo 'done'