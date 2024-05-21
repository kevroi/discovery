#!/bin/bash


# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Python virtual environment is not activated."
    echo "Please activate your virtual environment and try again."
    exit 1
fi

# Start a new detached tmux session
tmux new-session -d -s atari_session

# Create new windows and send commands
tmux new-window -t atari_session:1 -n 'Seaquest-relu1'
tmux send-keys -t atari_session:1 'python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Seaquest-v5" \
    --activation "relu" \
    --feat_dim 512 \
    --use_wandb' C-m

tmux new-window -t atari_session:2 -n 'Seaquest-relu2'
tmux send-keys -t atari_session:2 'python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Seaquest-v5" \
    --activation "relu" \
    --feat_dim 512 \
    --use_wandb' C-m

tmux new-window -t atari_session:3 -n 'Seaquest-fta1'
tmux send-keys -t atari_session:3 'python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Seaquest-v5" \
    --activation "fta" \
    --feat_dim 10240 \
    --use_wandb' C-m

tmux new-window -t atari_session:4 -n 'Seaquest-fta2'
tmux send-keys -t atari_session:4 'python3 discovery/experiments/FeatAct_atari/run_atari.py \
    --env_name "ALE/Seaquest-v5" \
    --activation "fta" \
    --feat_dim 10240 \
    --use_wandb' C-m

# Attach to the tmux session
tmux attach-session -t atari_session