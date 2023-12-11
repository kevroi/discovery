#!/bin/bash
#SBATCH --job-name=test_ppo
#SBATCH --account=rrg-whitem
#SBATCH --cpus-per-task=5
#SBATCH --time=0-02:55
#SBATCH --mem=6G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roice@ualberta.ca
#SBATCH -o /home/roice/scratch/discovery/logs/test.out # Standard output
#SBATCH -e /home/roice/scratch/discovery/logs/test.err # Standard error
module load python/3.9
source venv/bin/activate
python main.py