#!/bin/bash

# Loop 50 times
for i in {1..30}
do
    python3 discovery/experiments/FeatAct_climbing/run_climbing.py --use_wandb
done