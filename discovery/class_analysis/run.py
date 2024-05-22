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
from typing import Optional, Union

import argparse
import os
import numpy as np
import tqdm
import pickle
from discovery.utils import filesys
from discovery.class_analysis import lib, datasources, run_lib
from discovery.class_analysis.datatypes import (
    Setting,
    Data,
    EnvName,
    ModelType,
    AllTrainTestStats,
    BaseTrainTestStats,
)
from sklearn import random_projection


_RESULT_STORE = "discovery/class_analysis/results.pkl"
_WHITELIST_SETTINGS = None  # Do every setting.
_MAX_NUM_MODELS_TO_PROCESS = -1  # No limit.


# Some settings for debugging.
# _MAX_NUM_MODELS_TO_PROCESS = 2
# _MAX_NUM_MODELS_TO_PROCESS = -1
# _RESULT_STORE = "discovery/class_analysis/results_DEBUG2.pkl"
# _WHITELIST_SETTINGS = [
#     Setting(multitask=False, model_type=ModelType.CNN, env_name=EnvName.Seaquest)
# ]

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
    "--dry_run",
    action="store_true",  # set to false if we do not pass this argument
    help="Don't actually process the files, just print what would be done.",
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
parser.add_argument(
    "--analysis_type",
    type=str,
    choices=("minigrid_train", "minigrid_transfer", "seaquest"),
    required=True,
    help="Which version of the analysis to run.",
)


# To get the config from wandb, we need to know the project name. This is
# not currently encoded in the path, so we need to keep a mapping.
# TODO: test whether things still work with the //szepi/prefix on project names.
_PATH_PREFIX_TO_PROJECT_NAME = {
    "discovery/experiments/FeatAct_minigrid/models/single_task_fta/TwoRoomEnv/PPO": "//szepi/TwoRoomsSingleTask2",
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_multi_task_cnn/TwoRoomEnv/PPO": "//szepi/two_rooms_multi_task_cnn",
    "discovery/experiments/FeatAct_atari/models/PPO_ALE": "//szepi/PPO_on_Atari",
}

# fmt: off
_EXCLUDED_DIRECTORIES = [
    # Cannot have trailing slashes.

    # these are old models that we don't need to analyze anymore. They had all task locations.
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_multi_task_cnn/TwoRoomEnv/PPO",
    "discovery/experiments/FeatAct_minigrid/models/multi_task_fta/TwoRoomEnv/PPO",

    # Snapshot directories for Atari.
    "discovery/experiments/FeatAct_atari/models/dir_PPO_ALE",
    "discovery/experiments/FeatAct_minigrid/models/dir_PPO_TwoRoomEnv_wm1nfc2w",
]
# fmt: on

_PATH_PREFIX_TO_SETTING = {
    # "discovery/experiments/FeatAct_minigrid/models/multi_task_fta/TwoRoomEnv/PPO/": Setting(
    #     multitask=True, model_type=ModelType.FTA, env_name=EnvName.TwoRooms
    # ),
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_realmultitask_fta/TwoRoomEnv/PPO/": Setting(
        multitask=True, model_type=ModelType.FTA, env_name=EnvName.TwoRooms
    ),
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_realmultitask_cnn/TwoRoomEnv/PPO/": Setting(
        multitask=True, model_type=ModelType.CNN, env_name=EnvName.TwoRooms
    ),
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_single_task_cnn/TwoRoomEnv/PPO/": Setting(
        multitask=False, model_type=ModelType.CNN, env_name=EnvName.TwoRooms
    ),
    "discovery/experiments/FeatAct_atari/models/seaquest_cnn/": Setting(
        multitask=False, model_type=ModelType.CNN, env_name=EnvName.Seaquest
    ),
    "discovery/experiments/FeatAct_minigrid/models/two_rooms_newmultitask_cnn/TwoRoomEnv/PPO": Setting(
        multitask=True, model_type=ModelType.CNN, env_name=EnvName.TwoRooms
    ),
}

# TODO: this loads the seaquest data when we load the module, which takes long (10-20s)
# and is not necessary if we don't analyze seaquest models.
env_name_to_data_mgr_cls = {
    EnvName.TwoRooms: {
        "single_task": datasources.MiniGridData(),
        "transfer_train": datasources.MiniGridData([1, 2, 4, 6]),
        "transfer_test": datasources.MiniGridData([3, 5]),
    },
    EnvName.Seaquest: datasources.SeaquestData(
        obs_filepath=filesys.make_abs_path_in_root(
            "datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1).hdf5"
        ),
        label_filepath=filesys.make_abs_path_in_root(
            "datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1)_labels.npy",
        ),
        cutoff=1200,
    ),
}

# TODO: this is hardcoded for now.
env_name_to_feat_dims = {
    EnvName.TwoRooms: {ModelType.CNN: 32, ModelType.FTA: 640},
    EnvName.Seaquest: {ModelType.CNN: 512, ModelType.FTA: 10240},
}

wandb_api = wandb.Api()


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
    potential_prefixes_to_remove = [
        "PPO_TwoRoomEnv_",
        "Pong-v5_",
        "Seaquest-v5_",
        "MsPacman-v5_",
    ]
    for prefix in potential_prefixes_to_remove:
        if wandb_id.startswith(prefix):
            wandb_id = wandb_id[len(prefix) :]
            break

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
    project_name = project_name[0]
    if project_name.endswith("/"):
        project_name = project_name[:-1]
    assert project_name.startswith("//")
    run = wandb_api.run(f"{project_name}/{wandb_id}")
    setting = _setting_from_config(run.config)
    return setting, wandb_id


def _setting_from_config(config: dict) -> Setting:
    """Returns a Setting based on the wandb config of a run."""
    if config["env_name"] == "TwoRoomEnv":
        env_name = EnvName.TwoRooms
        multitask = config["random_hallway"]
    elif config["env_name"] == "ALE/Seaquest-v5":
        env_name = EnvName.Seaquest
        multitask = False
    else:
        raise ValueError(f"Unknown env_name: {config['env_name']}")
    if config["activation"] == "fta":
        model_type = ModelType.FTA
    else:
        model_type = ModelType.CNN
    return Setting(
        multitask=multitask,
        model_type=model_type,
        env_name=env_name,
    )


def process_model_at_path(
    model_path: str,
    file_name: str,
    all_results: Union[dict[Setting, Data], dict[Setting, AllTrainTestStats]],
    analysis_type: str,
    dry_run=False,
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

    if dry_run:
        print("    dry run -- would process this.")
        return None

    if incompatible_setting(setting.env_name, analysis_type):
        print("    skipping -- incompatible setting.")
        return None

    # So hacky..
    data_mgr = env_name_to_data_mgr_cls[setting.env_name]
    if isinstance(data_mgr, dict):
        # We can use any variant here..
        data_mgr = data_mgr["single_task"]

    obs_to_feats_fn = lib.get_obs_to_feats_fn(data_mgr.obs_preprocessor, model_path)
    gather_results(all_results, setting, wandb_id, obs_to_feats_fn, analysis_type)

    return setting


def gather_results(all_results, setting, wandb_id, obs_to_feats_fn, analysis_type):

    if analysis_type == "minigrid_train":
        results = lib.train_classifier_for_extractor(
            env_name_to_data_mgr_cls[setting.env_name]["single_task"], obs_to_feats_fn
        )
        print("    TRAIN linear acc:", results["linear"][0].acc)
        print("    TRAIN nonlin acc:", results["nonlinear"][0].acc)
        run_lib.append_or_create(all_results, setting, wandb_id, results)

    elif analysis_type == "seaquest":
        results = lib.train_classifier_for_extractor(
            env_name_to_data_mgr_cls[setting.env_name],
            obs_to_feats_fn,
            # test_size=0.00001,
            test_size=0.2,
        )
        print("    TRAIN linear acc:", results["linear"][0].acc)
        print("    TRAIN nonlin acc:", results["nonlinear"][0].acc)
        print("    TEST linear acc:", results["linear"][1].acc)
        print("    TEST nonlin acc:", results["nonlinear"][1].acc)
        cur_results = BaseTrainTestStats(
            lin_train=results["linear"][0],
            lin_test=results["linear"][1],
            nonlin_train=results["nonlinear"][0],
            nonlin_test=results["nonlinear"][1],
        )
        run_lib.append_or_create_V2(all_results, setting, wandb_id, cur_results)

    elif analysis_type == "minigrid_transfer":
        # if not setting.multitask:
        #     print("    skipping -- not a multitask model.")

        train_results = lib.train_classifier_for_extractor(
            env_name_to_data_mgr_cls[setting.env_name]["transfer_train"],
            obs_to_feats_fn,
        )
        print("    TRAIN linear acc:", train_results["linear"][0].acc)
        print("    TRAIN nonlin acc:", train_results["nonlinear"][0].acc)

        eval_data_source = env_name_to_data_mgr_cls[setting.env_name]["transfer_test"]
        transfer_results = {}
        for key, (_, _, details) in train_results.items():
            transfer_results[key] = lib.evaluate_extractor(
                eval_data_source, details["classifier"], details["obs_to_feats"]
            )
        print("    TRANSFER linear acc:", transfer_results["linear"].acc)
        print("    TRANSFER nonlin acc:", transfer_results["nonlinear"].acc)

        cur_results = BaseTrainTestStats(
            lin_train=train_results["linear"][0],
            lin_test=transfer_results["linear"],
            nonlin_train=train_results["nonlinear"][0],
            nonlin_test=transfer_results["nonlinear"],
        )
        run_lib.append_or_create_V2(all_results, setting, wandb_id, cur_results)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def main():
    args = parser.parse_args()
    print("Running with args:\n", args)
    print()
    print()

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
        all_results: Union[dict[Setting, Data], dict[Setting, AllTrainTestStats]] = {}
    else:
        all_results = run_lib.load_existing_results(args.result_path)

    # modified_settings = []
    num_new_settings = 0

    exclude_dirs = [filesys.make_abs_path_in_root(d) for d in _EXCLUDED_DIRECTORIES]
    abs_load_dir_path = os.path.abspath(args.load_dir)
    if args.recursive:
        all_files = run_lib.scantree(abs_load_dir_path, exclude_dirs)
    else:
        if abs_load_dir_path in exclude_dirs:
            print("The directory you provided is deprecated and will be skipped.")
            return
        all_files = os.scandir(abs_load_dir_path)

    skipped_due_to_error = []
    if abs_load_dir_path is not None:
        for dir_entry in tqdm.tqdm(list(all_files)):
            if dir_entry.is_dir():
                print("SKIPPING -- dir:", dir_entry.name)
                continue
            if dir_entry.is_file():
                if not dir_entry.name.endswith(".zip"):
                    print("SKIPPING -- not a .zip file:", dir_entry.name)
                    continue
                rel_path = run_lib.make_path_relative(dir_entry.path)
                try:
                    maybe_new_setting = process_model_at_path(
                        rel_path,
                        dir_entry.name,
                        all_results,
                        args.analysis_type,
                        dry_run=args.dry_run,
                    )
                except ValueError as e:
                    print("SKIPPING -- error: ", dir_entry.name)
                    skipped_due_to_error.append((rel_path, e))
                    # raise e
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

    added_rand_proj = add_random_projections(
        args.random_proj_seeds, all_results, args.analysis_type, dry_run=args.dry_run
    )

    added_obs = add_obserations(
        args.random_proj_seeds, all_results, args.analysis_type, dry_run=args.dry_run
    )

    added_new_runs = added_rand_proj or added_obs

    if added_new_runs and not args.dry_run:
        with open(args.result_path, "wb") as f:
            pickle.dump(all_results, f)

    print()
    print()
    print("Skipped due to error:")
    for path, e in skipped_due_to_error:
        print(" *", path)
        print("      ", e)


def add_random_projections(
    num_seeds: int,
    all_results: Union[dict[Setting, Data], dict[Setting, AllTrainTestStats]],
    analysis_type: str,
    dry_run: bool,
) -> bool:
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
            if dry_run:
                print("    dry run -- would process this.")
                continue

            gather_results(all_results, setting, None, obs_to_feats, analysis_type)

            # results = lib.train_classifier_for_extractor(
            #     env_name_to_data_mgr_cls[env],
            #     obs_to_feats,
            # )
            # # Reporting train stats only here.
            # print("    linear acc:", results["linear"][0].acc)
            # print("    nonlin acc:", results["nonlinear"][0].acc)
            # run_lib.append_or_create(all_results, setting, None, results)
        return True

    added_new_run = False
    for env in present_envs:
        if incompatible_setting(env, analysis_type):
            print(f"Skipping {env} due to incompatible analysis type: {analysis_type}.")

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
        s = Setting(
            # Multi-task is not relevant for random projections.
            multitask=True,
            env_name=env,
            model_type=ModelType.RANDOM_PROJ_GAUSS,
        )
        added_new_run = (
            run_proj(rand_projector_gauss, s)
            or added_new_run  # NOTE: this order is important.
        )
        print("Adding sparse random projections for", env)
        s = Setting(
            # Multi-task is not relevant for random projections.
            multitask=True,
            env_name=env,
            model_type=ModelType.RANDOM_PROJ_SPARSE,
        )
        added_new_run = (
            run_proj(rand_projector_sparse, s)
            or added_new_run  # NOTE: this order is important.
        )
    return added_new_run


def add_obserations(
    num_seeds: int,
    all_results: Union[dict[Setting, Data], dict[Setting, AllTrainTestStats]],
    analysis_type: str,
    dry_run: bool,
) -> bool:
    """Adds observations to the results; updates all_results but does not save."""
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
            batch_of_flat = np.stack(list_of_obs, axis=0).reshape(len(list_of_obs), -1)
            return proj(batch_of_flat)

        for seed in tqdm.trange(num_seeds_needed):
            if dry_run:
                print("    dry run -- would process this.")
                continue

            gather_results(all_results, setting, None, obs_to_feats, analysis_type)
        return True

    added_new_run = False
    for env in present_envs:
        if incompatible_setting(env, analysis_type):
            print(f"Skipping {env} due to incompatible analysis type: {analysis_type}.")

        if env not in env_name_to_feat_dims:
            print()
            print(f"Skipping {env} for now --- parameters not set correctly.")
            print()
            continue

        observation_map = lambda x: x

        print("Adding observations for", env)
        s = Setting(
            # Multi-task is not relevant for observations.
            multitask=True,
            env_name=env,
            model_type=ModelType.OBSERVATION,
        )
        added_new_run = (
            run_proj(observation_map, s)
            or added_new_run  # NOTE: this order is important.
        )
    return added_new_run


def incompatible_setting(env: EnvName, analysis_type: str) -> bool:
    if (
        analysis_type == "minigrid_train" or analysis_type == "minigrid_transfer"
    ) and env != EnvName.TwoRooms:
        return True

    if analysis_type == "seaquest" and env != EnvName.Seaquest:
        return True

    return False


if __name__ == "__main__":
    main()
