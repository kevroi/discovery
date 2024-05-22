#!/bin/bash
python discovery/class_analysis/run.py \
    --load_dir=discovery/experiments/FeatAct_minigrid/models/ \
    --analysis_type=minigrid_train \
    --recursive \
    --result_path=discovery/class_analysis/two_rooms_RESULTS_TRAIN.pkl \
    --random_proj_seeds=20

    # --random_proj_seeds=20 \
    # --ignore_existing

    # --ignore_existing \
    # --dry_run
