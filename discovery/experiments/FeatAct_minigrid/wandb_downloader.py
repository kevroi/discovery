import pandas as pd
import wandb

api = wandb.Api()
entity, project = "roice", "SubgoalFeats_CosSim"
metric = "rollout/ep_rew_mean"
# metric = "Cosine Similarity with phi_subgoal_0"
# metric = "Cosine Similarity with phi_subgoal_1"

# MongoDB-style query for your project
filter_dict = {
    "$and": [
        {"config.env_name": "MiniGrid-DoorKey-5x5-v0"},
        {"config.learner": "PPO"},
        {"config.activation": "crelu"},
        {"config.feat_dim": 40},
        # {"summary.rollout/ep_rew_mean": {"$gt": 0.95}}
    ]
}
runs = api.runs(f"{entity}/{project}", filters=filter_dict)
runs_df = pd.DataFrame()

for run in runs:
    runs_df[run.id] = run.history()[metric]

runs_df = runs_df.fillna(0.0)
# save to csv
runs_df.to_csv(
    f"experiments/FeatAct_minigrid/wandb_export_data/{project}_{filter_dict['$and'][0]['config.env_name']}_{filter_dict['$and'][1]['config.learner']}_{filter_dict['$and'][2]['config.activation']}_{filter_dict['$and'][3]['config.feat_dim']}_data.csv"
)
