import pandas as pd
import wandb

api = wandb.Api()
entity, project = "szepi", "PPO_on_Atari"
metric = "rollout/ep_rew_mean"
# metric = "Cosine Similarity with phi_subgoal_0"
# metric = "Cosine Similarity with phi_subgoal_1"

# MongoDB-style query for your project
filter_dict = {
    "$and": [
        {"config.env_name": "ALE/Seaquest-v5"},
        {"config.learner": "PPO"},
        {"config.activation": "fta"},
    ]
}
runs = api.runs(f"{entity}/{project}", filters=filter_dict)
runs_df = pd.DataFrame()

for run in runs:
    runs_df[run.id] = run.history()[metric]

runs_df = runs_df.fillna(0.0)
# save to csv
runs_df.to_csv(
    f"discovery/experiments/FeatAct_atari/wandb_export_data/{project}_{filter_dict['$and'][0]['config.env_name'][4:]}_{filter_dict['$and'][1]['config.learner']}_{filter_dict['$and'][2]['config.activation']}.csv"
)
