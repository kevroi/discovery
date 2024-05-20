#!/bin/bash

# When starting over ssh, use tmux, otherwise the jobs will stop
# when the terminal is closed.

# Loop 50 times
for i in {1..10}
do
    python3 discovery/experiments/FeatAct_minigrid/run_minigrid.py --device="cuda:2" --use_wandb &
done