import os
import random
import numpy as np
import yaml
from argparse import ArgumentParser
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO, DQN
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from cnn import MinigridFeaturesExtractor
import wandb

## HELPER FUNCTIONS ##
def make_env(config):
    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = RGBImgObsWrapper(env) # FullyObsWrapper runs faster locally, but uses ints instead of 256-bit RGB
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env
######################

def main(args):
    # Load YAML hyperparameters
    with open(f'experiments/FeatAct_minigrid/{args.config_file}', 'r') as f:
        hparam_yaml = yaml.safe_load(f)
    # Replace yaml default hypers with command line arguments
    for k, v in vars(args).items():
        if v is not None:
            hparam_yaml[k] = v
    
    # # Set random seed
    # np.random.seed(hparam_yaml['seed'])   # TODO: Check if this is the best way to set the seed
    # random.seed(hparam_yaml['seed'])

    # Setup logger if using wandb
    if hparam_yaml['use_wandb']:
        run = wandb.init(
            project=hparam_yaml['project_name'], 
            config=hparam_yaml,
            sync_tensorboard=True, 
            monitor_gym=True,
            save_code=True,
            )
        run_id = run.id
    else:
        run = None
        run_id = 'debug'
    
    # Create environment
    env = DummyVecEnv([lambda: make_env(config=hparam_yaml)]*hparam_yaml['n_envs'])
    if hparam_yaml['record_video']:
        env = VecVideoRecorder(
                                env,
                                f"videos/{run_id}",
                                record_video_trigger=lambda x: x % 100000 == 0,
                                video_length=200,
                            )
    
    # Create agent
    policy_kwargs = dict(
                        features_extractor_class=MinigridFeaturesExtractor,
                        features_extractor_kwargs=dict(features_dim=hparam_yaml['feat_dim']),
                        )
    if hparam_yaml["learner"] == "PPO":
        model = PPO(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    elif hparam_yaml["learner"] == "DQN":
        model = DQN(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    
    print(f"Training {hparam_yaml['learner']} on {hparam_yaml['env_name']} with {hparam_yaml['feat_dim']} features.")
    model.learn(total_timesteps=hparam_yaml["timesteps"])
    model.save(f"experiments/FeatAct_minigrid/models/{hparam_yaml['learner']}_{hparam_yaml['env_name']}")
        

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', 
        type=str, 
        default= 'config.yaml',
        help='Configuration file to use.'
    )
    parser.add_argument(
        '--env_name', 
        type=str, 
        default='MiniGrid-Empty-5x5-v0', 
        help='Minigrid environment official name.'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=3e-4, 
        help='Learning rate of the Adam optimizer used to optimise surrogate loss.'
    )
    parser.add_argument(
        '--use_wandb', 
        action='store_true', # set to false if we do not pass this argument
        help='Raise the flag to use wandb.'
    )
    args = parser.parse_args()
    main(args)