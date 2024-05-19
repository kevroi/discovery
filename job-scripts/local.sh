#!/bin/bash

# Loop 50 times
for i in {1..10}
do
    set -m; python3 discovery/experiments/FeatAct_minigrid/run_minigrid.py --use_wandb &
done