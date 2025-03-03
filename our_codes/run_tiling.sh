#!/bin/bash

#SBATCH --job-name=tiling      ## Name of the job
#SBATCH --output=run_tiling.out    ## Output file
#SBATCH --time=07:59:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8     ## The number of threads the code will use
#SBATCH --mem=100G     ## Real memory(MB) per CPU required by the job.
#SBATCH --gres=gpu:1


## Load the python interpreters

source /gladstone/finkbeiner/home/mahirwar/miniforge3/etc/profile.d/conda.sh
conda activate gigapath2
module load cuda/12.4

cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/prov-gigapath/our_codes

python3 tiling_slides.py 
