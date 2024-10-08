#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-2:59
#SBATCH --mail-type=ALL


echo $LEARNER, $ENV_NAME, $LR

module load python/3.10
source venv/bin/activate
wandb login

python $me/discovery/discovery/experiments/FeatAct_minigrid/run_minigrid.py \
    --env_name=$ENV_NAME --learner=$LEARNER --lr=$LR --use_wandb