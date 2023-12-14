#!/bin/bash
#SBATCH -J PPO_DK8_RGB
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-2:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roice@ualberta.ca
#SBATCH -o /home/roice/scratch/discovery/logs/PPO_DK8_RGB.out # Standard output
#SBATCH -e /home/roice/scratch/discovery/logs/PPO_DK8_RGB.err # Standard error

module load python/3.9
source venv/bin/activate
wandb login
python $me/discovery/experiments/FeatAct_minigrid/run_minigrid.py