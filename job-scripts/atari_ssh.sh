#!/bin/bash

set -m; python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Pong-v5" --activation "relu" --feat_dim 512 --use_wandb &
set -m; python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Seaquest-v5" --activation "relu" --feat_dim 512 --use_wandb &
set -m; python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/MsPacman-v5" --activation "relu" --feat_dim 512 --use_wandb &
set -m; python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Pong-v5" --activation "fta" --feat_dim 10240 --use_wandb &
set -m; python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Seaquest-v5" --activation "fta" --feat_dim 10240 --use_wandb &
set -m; python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/MsPacman-v5" --activation "fta" --feat_dim 10240 --use_wandb &
