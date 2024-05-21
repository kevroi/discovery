"""Analyze models in a directory for subgoalness.

Point this script to a directory and it will analyze all models
it can find in that directory for subgoalness. The script will
figure out what setting the model was trained in (e.g. the environment).

The script will accumulate results in the results file, over multiple calls.

A path may look like
discovery/experiments/FeatAct_minigrid/models/single_task_fta/TwoRoomEnv/PPO/3b4zmkze.zip,
or may look slightly different, though it will always end with a wandb run id string.

"""

import functools
from pyexpat import model
import wandb
from typing import Optional

import argparse
import os
import numpy as np
import tqdm
import pickle
from discovery.utils import filesys
from discovery.class_analysis import lib, datasources
from discovery.class_analysis.datatypes import Setting, Data, EnvName, ModelType
from sklearn import random_projection


_RESULT_STORE = "discovery/class_analysis/results.pkl"
_WHITELIST_SETTINGS = None  # Do every setting.
_MAX_NUM_MODELS_TO_PROCESS = -1  # No limit.


# Some settings for debugging.
# _MAX_NUM_MODELS_TO_PROCESS = 2
# _MAX_NUM_MODELS_TO_PROCESS = -1
# _RESULT_STORE = "discovery/class_analysis/results_DEBUG2.pkl"
_WHITELIST_SETTINGS = [
    Setting(multitask=False, model_type=ModelType.CNN, env_name=EnvName.Seaquest)
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_dir",
    type=str,
    help="The directory with the models to load. May be relative to project root.",
)
parser.add_argument(
    "--recursive",
    action="store_true",  # set to false if we do not pass this argument
    help="If true, will search recursively for models in the directory.",
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
    "discovery/experiments/FeatAct_atari/models/seaquest_cnn/": Setting(
        multitask=False, model_type=ModelType.CNN, env_name=EnvName.Seaquest
    ),
}

# TODO: this loads the seaquest data when we load the module, which takes long (10-20s)
# and is not necessary if we don't analyze seaquest models.
env_name_to_data_mgr_cls = {
    EnvName.TwoRooms: datasources.MiniGridData(),
    EnvName.Seaquest: datasources.SeaquestData(
        obs_filepath=filesys.make_abs_path_in_root(
            "datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1).hdf5"
        ),
        label_filepath=filesys.make_abs_path_in_root(
            "datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1)_labels.npy",
        ),
    ),
}

# TODO: this is hardcoded for now.
env_name_to_feat_dims = {
    EnvName.TwoRooms: {ModelType.CNN: 32, ModelType.FTA: 640},
    EnvName.Seaquest: {ModelType.CNN: 512, ModelType.FTA: 10240},
}

wandb_api = wandb.Api()


def load_existing_results(result_path: str):
    """Returns the existing results, or an empty dict if there are none."""
    try:
        with open(result_path, "rb") as f:
            existing_results = pickle.load(f)
    except FileNotFoundError:
        existing_results = {}
        print(
            "There is no results file at",
            os.path.join(os.getcwd(), result_path),
            "\nStarting a new results file!",
        )
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
    """Returns a Setting based on the wandb config of a run."""
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


def _append_or_create(all_results: dict, setting, wandb_id, cur_results):
    """Appends the results to the data for the setting, or creates a new entry."""
    lin_acc, lin_conf_mat, lin_details = cur_results["linear"]
    nonlin_acc, nonlin_conf_mat, nonlin_details = cur_results["nonlinear"]
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


def process_model_at_path(
    model_path: str, file_name: str, all_results: dict[Setting, Data]
) -> Optional[Setting]:
    """Updates all_results with an analysis of the model at model_path."""
    # See the comment at the top of the file for the format of the path.
    setting, wandb_id = extract_setting(model_path, file_name)
    if _WHITELIST_SETTINGS is not None and setting not in _WHITELIST_SETTINGS:
        return None

    print("PROCESSING:", model_path)
    if setting in all_results and wandb_id in all_results[setting].wandb_ids:
        print("    skipping -- already processed.")
        return None

    results = lib.process_saved_model(
        env_name_to_data_mgr_cls[setting.env_name], model_path
    )
    print("    linear acc:", results["linear"][0])
    print("    nonlin acc:", results["nonlinear"][0])
    _append_or_create(all_results, setting, wandb_id, results)
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

    if args.load_dir is None:
        print()
        print("No directory to load from (--load_dir) was provided.")
        print()

    if args.ignore_existing:
        # We pretend there were no existing results.
        all_results = {}
    else:
        all_results = load_existing_results(args.result_path)

    # modified_settings = []
    num_new_settings = 0

    if args.recursive:
        all_files = scantree(args.load_dir)
    else:
        all_files = os.scandir(args.load_dir)

    skipped_due_to_error = []
    if args.load_dir is not None:
        for dir_entry in tqdm.tqdm(list(all_files)):
            if dir_entry.is_dir():
                print("SKIPPING -- dir:", dir_entry.name)
                continue
            if dir_entry.is_file():
                if not dir_entry.name.endswith(".zip"):
                    print("SKIPPING -- not a .zip file:", dir_entry.name)
                    continue
                rel_path = _make_path_relative(dir_entry.path)
                try:
                    maybe_new_setting = process_model_at_path(
                        rel_path, dir_entry.name, all_results
                    )
                except ValueError as e:
                    print("SKIPPING -- error: ", dir_entry.name)
                    skipped_due_to_error.append((rel_path, e))
                    continue
                if maybe_new_setting is not None:
                    num_new_settings += 1
                    # We save right away if we have a new setting;
                    # this is to avoid losing data if the script crashes.
                    # For now this is very fast.
                    with open(args.result_path, "wb") as f:
                        pickle.dump(all_results, f)
                    if num_new_settings == _MAX_NUM_MODELS_TO_PROCESS:
                        break

    added_new_runs = add_random_projections(args.random_proj_seeds, all_results)

    if added_new_runs:
        with open(args.result_path, "wb") as f:
            pickle.dump(all_results, f)

    print()
    print()
    print("Skipped due to error:")
    for path, e in skipped_due_to_error:
        print(" *", path)
        print("      ", e)


def add_random_projections(num_seeds: int, all_results: dict[Setting, Data]) -> bool:
    """Adds random projections to the results; updates all_results but does not save."""
    present_envs = {s.env_name for s in all_results}

    def run_proj(proj, setting) -> bool:
        # Figure out how many seeds we want to run.
        print(setting)
        if setting in all_results:
            print(
                f"    already have {all_results[setting].num_runs} runs for this setting."
            )
            num_seeds_needed = max(0, num_seeds - all_results[setting].num_runs)
            print(f"    Need {num_seeds_needed} more.")
        else:
            num_seeds_needed = num_seeds
            print(f"    no runs for this setting yet, need {num_seeds_needed} more.")
        if num_seeds_needed <= 0:
            print("    skipping -- enough runs already.")
            return False

        def obs_to_feats(list_of_obs):
            # Calling proj.fit_transform generates a new random projection each time.
            # (If we wanted the same one repeatedly, we'd fit first, then each transform
            # would yield the same reults.)
            batch_of_flat = np.stack(list_of_obs, axis=0).reshape(len(list_of_obs), -1)
            return proj.fit_transform(batch_of_flat)

        for seed in tqdm.trange(num_seeds_needed):
            results = lib.process_model(
                env_name_to_data_mgr_cls[env],
                obs_to_feats,
            )
            print("    linear acc:", results["linear"][0])
            print("    nonlin acc:", results["nonlinear"][0])
            _append_or_create(all_results, setting, None, results)
        return True

    added_new_run = False
    for env in present_envs:
        if env not in env_name_to_feat_dims:
            print()
            print(f"Skipping {env} for now --- parameters not set correctly.")
            print()
            continue

        feat_dims = env_name_to_feat_dims[env]
        rand_projector_gauss = random_projection.GaussianRandomProjection(
            n_components=feat_dims[ModelType.CNN]
        )
        rand_projector_sparse = random_projection.SparseRandomProjection(
            n_components=feat_dims[ModelType.FTA]
        )

        print("Adding gaussian random projections for", env)
        added_new_run = (
            run_proj(
                rand_projector_gauss,
                Setting(
                    # Multi-task is not relevant for random projections.
                    multitask=True,
                    env_name=env,
                    model_type=ModelType.RANDOM_PROJ_GAUSS,
                ),
            )
            or added_new_run  # NOTE: this order is important.
        )
        print("Adding sparse random projections for", env)
        added_new_run = (
            run_proj(
                rand_projector_sparse,
                Setting(
                    # Multi-task is not relevant for random projections.
                    multitask=True,
                    env_name=env,
                    model_type=ModelType.RANDOM_PROJ_SPARSE,
                ),
            )
            or added_new_run  # NOTE: this order is important.
        )
    return added_new_run


def _make_path_relative(path: str) -> str:
    """Returns the path relative to the current working directory/project root."""
    # Not really tested.
    if path.startswith("/"):
        return path[len(os.getcwd()) + 1 :]
    return path


def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    # from https://stackoverflow.com/a/33135143
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry


if __name__ == "__main__":
    main()
