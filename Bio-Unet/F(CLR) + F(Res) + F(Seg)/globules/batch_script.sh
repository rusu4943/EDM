#!/bin/bash
#SBATCH --job-name=globules_07
#SBATCH --output=globules_07.log
#SBATCH --error=globules_07.log
#SBATCH --time=2-00:00:00  
#SBATCH --partition=gpu  
#SBATCH --cpus-per-task=31 
#SBATCH --mem-per-cpu=8GB 
#SBATCH --gres=gpu:a40:2  
#SBATCH --account=rostamim_919

source /spack/conda/miniconda3/23.3.1/bin/activate XAI

/home1/ruitongs/.conda/envs/XAI/bin/python SIMCLR+Resnet50+Unet.py
