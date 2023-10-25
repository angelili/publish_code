#!/bin/bash
#SBATCH --job-name=simulation
ray start --head
srun python3 main.py
