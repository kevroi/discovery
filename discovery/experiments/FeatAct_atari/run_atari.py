import os
import random
import numpy as np
import yaml
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3 import PPO, DQN
from discovery.agents.ddqn import DoubleDQN
from discovery.utils.feat_extractors import *
import wandb
from discovery.utils.activations import *
from discovery.experiments.FeatAct_minigrid.helpers import make_env
from stable_baselines3.common.env_util import make_atari_env
from discovery.utils.save_callback import SnapshotCallback


def main(args):
    # Load YAML hyperparameters
    with open(f"discovery/experiments/FeatAct_atari/{args.config_file}", "r") as f:
        hparam_yaml = yaml.safe_load(f)
    # Replace yaml default hypers with command line arguments
    for k, v in vars(args).items():
        if v is not None:
            hparam_yaml[k] = v
        else:
            print(f"Using default value for {k}: {hparam_yaml[k]}")

    # # Set random seed
    # np.random.seed(hparam_yaml['seed'])   # TODO: Check if this is the best way to set the seed
    # random.seed(hparam_yaml['seed'])

    # Setup logger if using wandb
    if hparam_yaml["use_wandb"]:
        run = wandb.init(
            project=hparam_yaml["project_name"],
            config=hparam_yaml,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        run_id = run.id
        wandb.run.log_code("./experiments/FeatAct_atari/")
    else:
        run = None
        run_id = "debug"

    # Create environment
    env = make_atari_env(hparam_yaml["env_name"], n_envs=hparam_yaml["n_envs"], seed=0)
    if hparam_yaml["frame_stack"] > 1:
        env = VecFrameStack(env, n_stack=hparam_yaml["frame_stack"])
    if hparam_yaml["record_video"]:
        env = VecVideoRecorder(
            env,
            f"videos/{run_id}",
            record_video_trigger=lambda x: x % 100000 == 0,
            video_length=200,
        )

    # Create agent
    if hparam_yaml["feat_extractor"] == "cnn":
        policy_kwargs = dict(
            features_extractor_class=NatureCNN,
            features_extractor_kwargs=dict(
                features_dim=hparam_yaml["feat_dim"],
                last_layer_activation=hparam_yaml["activation"],
            ),
        )
    else:
        raise ValueError("Invalid CNN type. Choose from: 'cnn'.")

    if hparam_yaml["learner"] == "PPO":
        model = PPO(
            hparam_yaml["policy_type"],
            env,
            learning_rate=hparam_yaml["lr"],
            gamma=hparam_yaml["gamma"],
            batch_size=hparam_yaml["batch_size"],
            n_epochs=hparam_yaml["n_epochs"],
            n_steps=hparam_yaml["n_steps"],
            stats_window_size=hparam_yaml["stats_window_size"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"runs/{run_id}",
        )
        if hparam_yaml["use_wandb"]:
            wandb.watch(
                model.policy.features_extractor, log_freq=100, log="all", log_graph=True
            )
    elif hparam_yaml["learner"] == "DQN":
        model = DQN(
            hparam_yaml["policy_type"],
            env,
            learning_rate=hparam_yaml["lr"],
            gamma=hparam_yaml["gamma"],
            batch_size=hparam_yaml["batch_size"],
            learning_starts=hparam_yaml["learning_starts"],
            train_freq=hparam_yaml["train_freq"],
            exploration_final_eps=hparam_yaml["exploration_final_eps"],
            target_update_interval=hparam_yaml["target_update_interval"],
            stats_window_size=hparam_yaml["stats_window_size"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"runs/{run_id}",
        )
        if hparam_yaml["use_wandb"]:
            wandb.watch(
                model.policy.q_net.features_extractor,
                log_freq=100,
                log="all",
                log_graph=True,
            )
    elif hparam_yaml["learner"] == "DDQN":
        model = DoubleDQN(
            hparam_yaml["policy_type"],
            env,
            learning_rate=hparam_yaml["lr"],
            gamma=hparam_yaml["gamma"],
            batch_size=hparam_yaml["batch_size"],
            learning_starts=hparam_yaml["learning_starts"],
            train_freq=hparam_yaml["train_freq"],
            exploration_final_eps=hparam_yaml["exploration_final_eps"],
            target_update_interval=hparam_yaml["target_update_interval"],
            stats_window_size=hparam_yaml["stats_window_size"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"runs/{run_id}",
        )
        if hparam_yaml["use_wandb"]:
            wandb.watch(
                model.policy.q_net.features_extractor,
                log_freq=100,
                log="all",
                log_graph=True,
            )

    filename_id = (
        f"{hparam_yaml['project_name']}/{hparam_yaml['env_name']}/"
        f"{hparam_yaml['learner']}/{run_id}"
    )
    snapshot_dir = f"discovery/experiments/FeatAct_atari/model_snapshots/{filename_id}"
    snapshot_callback = SnapshotCallback(
        check_freq=500_000 // hparam_yaml["n_envs"],
        log_dir=snapshot_dir,
        verbose=1,
    )

    print(
        f"Training {hparam_yaml['learner']} on {hparam_yaml['env_name']} with {hparam_yaml['feat_dim']} features."
    )
    model.learn(total_timesteps=hparam_yaml["timesteps"], callback=snapshot_callback)
    save_loc = f"discovery/experiments/FeatAct_atari/models/{filename_id}"
    model.save(save_loc)
    print(f"Model saved at {save_loc}")

    # if hparam_yaml["analyse_rep"]:
    #     from analyse_rep import get_feats

    # feature_activations = get_feats(model, hparam_yaml)
    # feature_activations = get_feats(model, hparam_yaml, see_bad_obs=True) # Uncomment to analyse bad observations too


if __name__ == "__main__":

    os.environ["WANDB__SERVICE_WAIT"] = "300"  # Waiting time for wandb to start

    parser = ArgumentParser()
    # The config file and things we want to sweep over (overwriting the config file)
    # CAREFUL: These default args also overwrite the config file
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Configuration file to use.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Minigrid environment official name.",
    )
    parser.add_argument(
        "--learner", type=str, default=None, help="Learning algorithm used."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function [LIST CHOICES].",
    )
    parser.add_argument(
        "--feat_dim", type=int, default=None, help="Dimension of feature space."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate of the Adam optimizer used to optimise surrogate loss.",
    )
    parser.add_argument(
        "--run_num",
        type=int,
        default=-1,
        help="Number of the run for multirun experiments.",
    )
    parser.add_argument(
        "--analyse_rep",
        action="store_true",  # set to false if we do not pass this argument
        help="Raise the flag to analyse feature vector.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",  # set to false if we do not pass this argument
        help="Raise the flag to use wandb.",
    )
    args = parser.parse_args()
    main(args)
