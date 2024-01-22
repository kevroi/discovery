#!/bin/bash
#SBATCH --job-name=PPO_MontRev_WB
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-2:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roice@ualberta.ca
#SBATCH -o /home/roice/scratch/discovery/logs/%x.out
#SBATCH -e /home/roice/scratch/discovery/logs/%x.err

module load python/3.9
source venv/bin/activate
wandb login
python $me/discovery/experiments/FeatAct_minigrid/run_minigrid.py