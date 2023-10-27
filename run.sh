#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --time=24:00:00 #the maximum walltime in format h:m:s
#SBATCH --nodelist=n01
#SBATCH --exclusive # Request exclusive access to the node
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL
ray start --address='10.10.21.21:6379'
srun python3 main.py
