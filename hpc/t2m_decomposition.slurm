#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72 #18
#SBATCH --gpus=4 #1
#SBATCH --partition=gpu
#SBATCH --mem=120G
#SBATCH --error=/home/thappe/HeaT/outs/job%j.err
#SBATCH --output=/home/thappe/HeaT/outs/job%j.out
#SBATCH --time=08:00:00
 
# Execute program 
python -u t2m_decomposition_ridge.py 
