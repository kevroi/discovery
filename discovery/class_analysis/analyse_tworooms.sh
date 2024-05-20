#!/bin/bash
python discovery/class_analysis/run.py \
    --load_dir=discovery/experiments/FeatAct_minigrid/models/ \
    --recursive \
    --result_path=discovery/class_analysis/two_rooms_RESULTS.pkl \
    --random_proj_seeds=20
