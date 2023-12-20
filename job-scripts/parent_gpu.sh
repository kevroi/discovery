#!/bin/bash
#SBATCH --job-name=parent_job
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-2:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roice@ualberta.ca
#SBATCH -o /home/roice/scratch/discovery/logs/%x.out
#SBATCH -e /home/roice/scratch/discovery/logs/%x.err

# Define your parameter sweep values
env_name='MiniGrid-Empty-8x8-v0' 
lrs=(0.1)
num_runs=10

# Loop over the parameter sweep values
for lr in ${lrs[@]}; do
    for i in $(seq 1 $num_runs); do
        # Submit a job for each parameter combination
        # sbatch --export=ALL,env_name="$env_name",lr="$lr" job-scripts/child_gpu.sh
        echo "sbatch --export=ALL,env_name=$env_name,lr=$lr job-scripts/child_gpu.sh"
    done
done
