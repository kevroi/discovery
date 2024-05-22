#!/bin/bash
python discovery/class_analysis/run.py \
    --load_dir=discovery/experiments/FeatAct_atari/models/ \
    --analysis_type=seaquest \
    --recursive \
    --result_path=discovery/class_analysis/atari_RESULTS.pkl \
    --random_proj_seeds=20
 
    # --random_proj_seeds=20
    # --random_proj_seeds=1 \
    # --ignore_existing

    # --ignore_existing \
    # --dry_run
