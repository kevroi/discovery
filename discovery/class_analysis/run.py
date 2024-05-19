"""Analyze models in a directory for subgoalness.

Point this script to a directory and it will analyze all models
it can find in that directory for subgoalness. The script will
figure out what setting the model was trained in (e.g. the environment).

The script will accumulate results in the results file, over multiple calls.

A path may look like
discovery/experiments/FeatAct_minigrid/models/single_task_fta/TwoRoomEnv/PPO/3b4zmkze.zip,
or may look slightly different, though it will always end with a wandb run id string.

"""

from ast import mod
import functools
from multiprocessing import process
from typing import Optional

import argparse
import os
from dataclasses import dataclass
import numpy as np
import tqdm
import pickle
from discovery.utils import filesys
from discovery.utils import sg_detection
from discovery.class_analysis import lib

from stable_baselines3 import PPO


_RESULT_STORE = "discovery/class_analysis/results.pkl"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_dir",
    type=str,
    required=True,
    help="The directory with the models to load. May be relative to root.",
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
    help="Where to store results. May be relative to root.",
)


@dataclass
class Setting:
    multitask: bool
    model_type: str
    env_name: str


@dataclass
class Data:
    wandb_ids: list[str]
    accuracies: list[float]
    conf_matrices: list[np.array]
    num_runs: int
    acc_mean: float
    acc_std_err: float


env_name_to_data_mgr_cls = {
    "TwoRoomEnv": lib.MiniGridData,
}


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
    print(name)
    return Setting(False, "cnn", "TwoRoomEnv"), name


def try_process_file(
    path: str, name: str, all_results: dict[Setting, Data]
) -> Optional[Setting]:
    """Updates all_results with an analysis of the model at path."""
    # See the comment at the top of the file for the format of the path.
    setting, wandb_id = extract_setting(path, name)
    return None
    # if setting in all_results and wandb_id in all_results[setting].wandb_ids:
    #     print("SKIPPING -- already processed:", name)
    #     return None

    # acc, conf_mat, details = lib.process_model(
    #     env_name_to_data_mgr_cls[setting.env_name], path
    # )

    # if setting in all_results:
    #     data = all_results[setting]
    #     data.wandb_ids.append(wandb_id)
    #     data.accuracies.append(acc)
    #     data.conf_matrices.append(conf_mat)
    #     data.num_runs += 1
    #     # Will update mean and std_err later.
    # else:
    #     all_results[setting] = Data(
    #         wandb_ids=[wandb_id],
    #         accuracies=[acc],
    #         conf_matrices=[conf_mat],
    #         num_runs=1,
    #         acc_mean=0.0,
    #         acc_std_err=0.0,
    #     )
    # return setting


def recalculate_stats(
    all_results: dict[Setting, Data], modified_settings: list[Setting]
):
    """Updates all_results based on the modified_settings list."""
    for setting in modified_settings:
        data = all_results[setting]
        data.acc_mean = np.mean(data.accuracies)
        data.acc_std_err = np.std(data.accuracies) / np.sqrt(data.num_runs)


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

    modified_settings = []
    for dir_entry in tqdm.tqdm(os.scandir(args.load_dir)):
        if dir_entry.is_dir():
            print("SKIPPING -- dir", dir_entry.name)
            continue
        if dir_entry.is_file():
            maybe_new_setting = try_process_file(
                dir_entry.path, dir_entry.name, all_results
            )
            if maybe_new_setting is not None:
                modified_settings.append(maybe_new_setting)

    assert not modified_settings, "No new settings to process."

    # if modified_settings:
    #     recalculate_stats(all_results, modified_settings)

    #     print("Saving results to", args.result_path)
    #     with open(args.result_path, "wb") as f:
    #         pickle.dump(all_results, f)
    # else:
    #     print("No new results to save.")


# minigrid_models = {
#     "multitask_cnn": Model(
#         multitask=True,
#         model_type="cnn",
#         wandb_id="5i6lt53x",
#         model_path="experiments/FeatAct_minigrid/models/PPO_TwoRoomEnv_5i6lt53x.zip",
#     ),
# }

# climbing_models = {}

# atari_models = {}


if __name__ == "__main__":
    main()
