import os
import random
import numpy as np
import yaml
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO, DQN
from discovery.agents.ddqn import DoubleDQN
from discovery.utils.feat_extractors import ClimbingFeatureExtractor
import wandb
from discovery.utils.activations import *
from discovery.experiments.FeatAct_minigrid.helpers import make_env

def main(args):
    # Load YAML hyperparameters
    with open(f'discovery/experiments/FeatAct_climbing/{args.config_file}', 'r') as f:
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
        wandb.run.log_code("./discovery/experiments/FeatAct_climbing/")
    else:
        run = None
        run_id = 'debug'
    
    # Create environment
    env = DummyVecEnv([lambda: make_env(config=hparam_yaml)]*hparam_yaml['n_envs'])
    
    # Create agent
    policy_kwargs = dict(
                    features_extractor_class=ClimbingFeatureExtractor,
                    features_extractor_kwargs=dict(),
                    )
    
    if hparam_yaml["learner"] == "PPO":
        model = PPO(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                gamma=hparam_yaml["gamma"],
                batch_size=hparam_yaml["batch_size"],
                n_epochs=hparam_yaml["n_epochs"],
                n_steps=hparam_yaml["n_steps"],
                stats_window_size=hparam_yaml["stats_window_size"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    elif hparam_yaml["learner"] == "DQN":
        model = DQN(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                gamma=hparam_yaml["gamma"],
                batch_size=hparam_yaml["batch_size"],
                learning_starts=hparam_yaml["learning_starts"],
                train_freq=hparam_yaml["train_freq"],
                exploration_final_eps=hparam_yaml["exploration_final_eps"],
                target_update_interval=hparam_yaml["target_update_interval"],
                stats_window_size=hparam_yaml["stats_window_size"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    elif hparam_yaml["learner"] == "DDQN":
        model = DoubleDQN(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                gamma=hparam_yaml["gamma"],
                batch_size=hparam_yaml["batch_size"],
                learning_starts=hparam_yaml["learning_starts"],
                train_freq=hparam_yaml["train_freq"],
                exploration_final_eps=hparam_yaml["exploration_final_eps"],
                target_update_interval=hparam_yaml["target_update_interval"],
                stats_window_size=hparam_yaml["stats_window_size"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    
    print(f"Training {hparam_yaml['learner']} on {hparam_yaml['env_name']}.")
    model.learn(total_timesteps=hparam_yaml["timesteps"])
    model.save(f"experiments/FeatAct_minigrid/models/{hparam_yaml['learner']}_{hparam_yaml['env_name']}_{run_id}")
    print(f"Model saved at experiments/FeatAct_minigrid/models/{hparam_yaml['learner']}_{hparam_yaml['env_name']}_{run_id}")

    # if hparam_yaml['analyse_rep']:
    #     from analyse_rep import get_feats
    #     feature_activations = get_feats(model, hparam_yaml)
    #     # feature_activations = get_feats(model, hparam_yaml, see_bad_obs=True) # Uncomment to analyse bad observations too
        

if __name__ == '__main__':

    os.environ["WANDB__SERVICE_WAIT"] = "300" # Waiting time for wandb to start

    parser = ArgumentParser()
    # The config file and things we want to sweep over (overwriting the config file)
    # CAREFUL: These default args also overwrite the config file
    parser.add_argument(
        '--config_file', 
        type=str, 
        default= 'config.yaml',
        help='Configuration file to use.'
    )
    parser.add_argument(
        '--env_name', 
        type=str, 
        default='discovery/Climbing-v0', 
        help='Environment registered name.'
    )
    parser.add_argument(
        '--learner', 
        type=str, 
        default='PPO', 
        help='Learning algorithm used.'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=3e-4, 
        help='Learning rate of the Adam optimizer used to optimise surrogate loss.'
    )
    parser.add_argument(
        '--run_num', 
        type=int,
        default=-1,
        help='Number of the run for multirun experiments.'
    )
    parser.add_argument(
        '--use_wandb', 
        action='store_true', # set to false if we do not pass this argument
        help='Raise the flag to use wandb.'
    )
    args = parser.parse_args()
    main(args)