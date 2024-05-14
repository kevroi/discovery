import os
import random
import numpy as np
import yaml
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import wandb


## HELPER FUNCTIONS ##
def make_env(config):
    env = make_atari_env(
        config["env_name"],
        n_envs=config["n_envs"],
        #  seed=config['env_seed']
    )
    env = VecFrameStack(env, n_stack=4)
    return env


######################

# config = {
#     "env_name": "ALE/MontezumaRevenge-v5",
#     "n_envs": 1, # TODO check if this means the same as PPO n_envs
#     "env_seed": 0,
#     "policy_type": "CnnPolicy",
#     "timesteps": 1e5,
#     "record_video": False,
# }

# run = wandb.init(
#                 project="PPO_on_Atari_Tests",
#                 config=config,
#                 sync_tensorboard=True,
#                 # monitor_gym=True,
#                 save_code=True,
#                 )

# env = make_env()
# print("env made")
# if config["record_video"]:
#     env = VecVideoRecorder(
#                             env,
#                             f"videos/{run.id}",
#                             record_video_trigger=lambda x: x % 100000 == 0,
#                             video_length=200,
#                         )

# model = PPO(config["policy_type"], env, verbose=1)
# print("agent made")
# model.learn(total_timesteps=config["timesteps"],
#             )
# print("learning done")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=False)
#     obs, rewards, dones, info = env.step(action)
#     print(rewards)
#     env.render("human")


def main(args):
    # Load hyperparameters from yaml file
    with open(f"experiments/FeatAct_atari/{args.config_file}", "r") as f:
        hparam_yaml = yaml.safe_load(f)
    # Replace yaml default hypers with command line arguments
    for k, v in vars(args).items():
        if v is not None:
            hparam_yaml[k] = v

    # Set seed
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
    else:
        run = None
        run_id = "debug"

    # Create environment
    env = make_env(config=hparam_yaml)
    if hparam_yaml["record_video"]:
        env = VecVideoRecorder(
            env,
            f"videos/{run_id}",
            record_video_trigger=lambda x: x % 100000 == 0,
            video_length=200,
        )

    # Create agent
    model = PPO(
        hparam_yaml["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run_id}"
    )
    model.learn(total_timesteps=hparam_yaml["timesteps"])
    model.save(f'experiments/FeatAct_atari/models/ppo_{hparam_yaml["env_name"]}')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Configuration file to use.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="ALE/MontezumaRevenge-v5",
        help="Atari environment official name.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate of the Adam optimizer used to optimise surrogate loss.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",  # set to false if we do not pass this argument
        help="Raise the flag to use wandb.",
    )
    args = parser.parse_args()
    main(args)
