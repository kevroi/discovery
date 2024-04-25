#!/bin/bash

compute_cluster="cedar"
env="TwoRoomEnv"
agent="PPO"
run_ids=(   "17lkjtmc"
            "2snf0e5o"
        )

for run_id in ${run_ids[@]}; do
    src_path="$compute_cluster://home/roice/projects/def-whitem/roice/discovery/experiments/FeatAct_minigrid/models/${agent}_${env}_${run_id}.zip"
    dest_path="./experiments/FeatAct_minigrid/models/"

    scp -r $src_path $dest_path
done
