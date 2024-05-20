"""Analyze models in a directory for subgoalness.

Point this script to a directory and it will analyze all models
it can find in that directory for subgoalness. The script will
figure out what setting the model was trained in (e.g. the environment).

The script will accumulate results in the results file, over multiple calls.

A path may look like
discovery/experiments/FeatAct_minigrid/models/single_task_fta/TwoRoomEnv/PPO/3b4zmkze.zip,
or may look slightly different, though it will always end with a wandb run id string.

"""

import wandb
from typing import Optional

import argparse
import os
import numpy as np
import tqdm
import pickle
from discovery.utils import filesys
from discovery.class_analysis import lib
from discovery.class_analysis.datatypes import Setting, Data, EnvName, ModelType


_RESULT_STORE = "discovery/class_analysis/results.pkl"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_dir",
    type=str,
    required=True,
    help="The directory with the models to load. May be relative to project root.",
)
parser.add_argument(
    "--ignore_existing",
    action="store_true",  # set to false if we do not pass this argument
    help="Effectively clears all previous results!",
)
parser.add_argument(
    "--result_path",
    type=str,
    default=_RESULT_STORE,
    help="Where to store results. May be relative to project root.",
)
parser.add_argument(
    "--random_proj_seeds",
    type=int,
    default=0,
    help="The minimum number of random projections analyzed for each environment.",
)


# To get the config from wandb, we need to know the project name. This is
# not currently encoded in the path, so we need to keep a mapping.
_PATH_PREFIX_TO_PROJECT_NAME = {
    "discovery/experiments/FeatAct_minigrid/models/single_task_fta/TwoRoomEnv/PPO/": "TwoRoomsSingleTask2",
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_multi_task_cnn/TwoRoomEnv/PPO/": "two_rooms_multi_task_cnn",
}

_PATH_PREFIX_TO_SETTING = {
    "discovery/experiments/FeatAct_minigrid/models/multi_task_fta/TwoRoomEnv/PPO/": Setting(
        multitask=True, model_type=ModelType.FTA, env_name=EnvName.TwoRooms
    ),
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_single_task_cnn/TwoRoomEnv/PPO/": Setting(
        multitask=False, model_type=ModelType.CNN, env_name=EnvName.TwoRooms
    ),
}


env_name_to_data_mgr_cls = {
    EnvName.TwoRooms: lib.MiniGridData,
}


wandb_api = wandb.Api()


def load_existing_results(result_path: str):
    try:
        with open(result_path, "rb") as f:
            existing_results = pickle.load(f)
    except FileNotFoundError:
        existing_results = {}
        print("There is no results file at", os.path.join(os.getcwd(), result_path))
    return existing_results


def extract_setting(path: str, name: str) -> tuple[Setting, str]:
    """Returns the setting and wandb_id based on the path."""
    # Example path:
    # discovery/experiments/FeatAct_minigrid/models/single_task_fta/TwoRoomEnv/PPO/3b4zmkze.zip,
    # or may look slightly different, though it will always end with a wandb run id string.
    # sooo turns out sometimes the filename is actually something like
    # `PPO_TwoRoomEnv_4nnytzzm.zip`.. for now we'll hardcode this case.
    parts = name.split(".")
    if len(parts) != 2:
        raise ValueError(f"Unexpected name: {name}; expected wandb_id.zip")
    wandb_id = parts[0]
    if wandb_id.startswith("PPO_TwoRoomEnv_"):
        wandb_id = wandb_id[len("PPO_TwoRoomEnv_") :]
    # First we check if the path is mapped to a setting explictly.
    setting = [s for p, s in _PATH_PREFIX_TO_SETTING.items() if path.startswith(p)]
    if setting:
        return setting[0], wandb_id
    # If not, we try to infer the wandb project name from the path.
    project_name = [
        n for p, n in _PATH_PREFIX_TO_PROJECT_NAME.items() if path.startswith(p)
    ]
    if not project_name:
        raise ValueError(f"Unknown project name for path: {path}")
    run = wandb_api.run(f"//{project_name[0]}/{wandb_id}")
    setting = _setting_from_config(run.config)
    return setting, wandb_id


def _setting_from_config(config: dict) -> Setting:
    if config["env_name"] == "TwoRoomEnv":
        env_name = EnvName.TwoRooms
        multitask = config["random_hallway"]
    else:
        raise NotImplementedError(f"Unknown env_name: {env_name}")
    if config["activation"] == "fta":
        model_type = ModelType.FTA
    else:
        model_type = ModelType.CNN
    return Setting(
        multitask=multitask,
        model_type=model_type,
        env_name=env_name,
    )


def try_process_file(
    path: str, name: str, all_results: dict[Setting, Data]
) -> Optional[Setting]:
    """Updates all_results with an analysis of the model at path."""
    # See the comment at the top of the file for the format of the path.
    setting, wandb_id = extract_setting(path, name)
    print("PROCESSING:", path)
    if setting in all_results and wandb_id in all_results[setting].wandb_ids:
        print("    skipping -- already processed.")
        return None

    results = lib.process_model(env_name_to_data_mgr_cls[setting.env_name], path)
    lin_acc, lin_conf_mat, lin_details = results["linear"]
    nonlin_acc, nonlin_conf_mat, nonlin_details = results["nonlinear"]
    print("    linear acc:", lin_acc)
    print("    nonlin acc:", nonlin_acc)

    if setting in all_results:
        data = all_results[setting]
        data.num_runs += 1
        data.wandb_ids.append(wandb_id)
        data.lin_accuracies.append(lin_acc)
        data.lin_conf_matrices.append(lin_conf_mat)
        data.lin_acc_mean = np.mean(data.lin_accuracies)
        data.lin_acc_std_err = np.std(data.lin_accuracies) / np.sqrt(data.num_runs)
        data.nonlin_accuracies.append(nonlin_acc)
        data.nonlin_conf_matrices.append(nonlin_conf_mat)
        data.nonlin_acc_mean = np.mean(data.nonlin_accuracies)
        data.nonlin_acc_std_err = np.std(data.nonlin_accuracies) / np.sqrt(
            data.num_runs
        )
    else:
        all_results[setting] = Data(
            wandb_ids=[wandb_id],
            num_runs=1,
            lin_accuracies=[lin_acc],
            lin_conf_matrices=[lin_conf_mat],
            lin_acc_mean=lin_acc,
            lin_acc_std_err=0.0,
            nonlin_accuracies=[nonlin_acc],
            nonlin_conf_matrices=[nonlin_conf_mat],
            nonlin_acc_mean=nonlin_acc,
            nonlin_acc_std_err=0.0,
        )

    return setting


def main():
    args = parser.parse_args()

    filesys.set_directory_in_project()

    # The strategy of this script:
    #
    # 1. The results file contains a dict from Setting to Data.
    # 2. We read these in and we may add to it.
    # 3. For each model file in the directory we find out the setting
    #    from the directory name or from wandb, then
    # 4. We run the subgoalness analysis. (Skip runs we already analyzed.)
    # 5. Store the new accumulated data.

    if args.ignore_existing:
        # We pretend there were no existing results.
        all_results = {}
    else:
        all_results = load_existing_results(args.result_path)

    # modified_settings = []
    for dir_entry in tqdm.tqdm(list(os.scandir(args.load_dir))):
        if dir_entry.is_dir():
            print("SKIPPING -- dir", dir_entry.name)
            continue
        if dir_entry.is_file():
            rel_path = _make_path_relative(dir_entry.path)
            maybe_new_setting = try_process_file(rel_path, dir_entry.name, all_results)
            if maybe_new_setting is not None:
                # We save right away if we have a new setting;
                # this is to avoid losing data if the script crashes.
                # For now this is very fast.
                with open(args.result_path, "wb") as f:
                    pickle.dump(all_results, f)

    # if modified_settings:
    #     recalculate_stats(all_results, modified_settings)

    #     print("Saving results to", args.result_path)
    #     with open(args.result_path, "wb") as f:
    #         pickle.dump(all_results, f)
    # else:
    #     print("No new results to save.")


def _make_path_relative(path: str) -> str:
    # Not really tested.
    if path.startswith("/"):
        return path[len(os.getcwd()) + 1 :]
    return path


if __name__ == "__main__":
    main()
