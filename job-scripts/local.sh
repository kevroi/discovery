#!/bin/bash

# Loop 50 times
for i in {1..20}
do
    python3 discovery/experiments/FeatAct_climbing/run_climbing.py --use_wandb
done