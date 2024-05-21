#!/bin/bash
python discovery/class_analysis/run.py \
    --load_dir=discovery/experiments/FeatAct_minigrid/models/two_rooms_newmultitask_cnn/TwoRoomEnv/PPO \
    --recursive \
    --result_path=discovery/class_analysis/two_rooms_RESULTS.pkl \
    --random_proj_seeds=2 \
    --ignore_existing 

    # --load_dir=discovery/experiments/FeatAct_minigrid/models/ \
    # --recursive \
    # --result_path=discovery/class_analysis/two_rooms_RESULTS.pkl \
    # --random_proj_seeds=20 \
    # --ignore_existing 
