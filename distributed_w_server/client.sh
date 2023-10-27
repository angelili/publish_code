#!/bin/bash
#SBATCH --job-name=client
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --time=24:00:00 #the maximum walltime in format h:m:s
#SBATCH --exclusive
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL

srun python3 client.py