#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18 
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --mem=120G
# #SBATCH --error=/home/thappe/HeaT/outs/job%j.err
# #SBATCH --output=/home/thappe/HeaT/outs/job%j.out
#SBATCH --time=24:00:00


module load 2022
module load IPython/8.5.0-GCCcore-11.3.0
module load JupyterHub/3.0.0-GCCcore-11.3.0

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load Cartopy/0.20.3-foss-2022a
module load scikit-build/0.11.1-foss-2022a

# Execute program 
python -u t2m_decomposition_ridge.py 
