#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-2:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roice@ualberta.ca
#SBATCH -o /home/roice/scratch/discovery/logs/%j.out
#SBATCH -e /home/roice/scratch/discovery/logs/%j.err


echo $LEARNER, $ENV_NAME, $FEAT_DIM, $LR, $RUN

module load python/3.9
source venv/bin/activate
wandb login

# python $me/discovery/discovery/experiments/FeatAct_atari/run_atari.py --env_name=$ENV_NAME --lr=$LR --use_wandb
python $me/discovery/discovery/experiments/FeatAct_minigrid/run_minigrid.py --env_name=$ENV_NAME --learner=$LEARNER --feat_dim=$FEAT_DIM --lr=$LR --run_num=$RUN --analyse_rep --use_wandb