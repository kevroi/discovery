import os
import random
import numpy as np
import yaml
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO, DQN
from agents.ddqn import DoubleDQN
from cnn import MinigridFeaturesExtractor, NatureCNN
import wandb
from utils import make_env

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
    if hparam_yaml["cnn"] == "nature":
        policy_kwargs = dict(
                            features_extractor_class=NatureCNN,
                            features_extractor_kwargs=dict(features_dim=hparam_yaml['feat_dim']),
                            )
    elif hparam_yaml["cnn"] == "minigrid":
        policy_kwargs = dict(
                            features_extractor_class=MinigridFeaturesExtractor,
                            features_extractor_kwargs=dict(features_dim=hparam_yaml['feat_dim']),
                            )
    
    if hparam_yaml["learner"] == "PPO":
        model = PPO(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                batch_size=hparam_yaml["batch_size"],
                n_epochs=hparam_yaml["n_epochs"],
                n_steps=hparam_yaml["n_steps"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    elif hparam_yaml["learner"] == "DQN":
        model = DQN(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                batch_size=hparam_yaml["batch_size"],
                learning_starts=hparam_yaml["learning_starts"],
                train_freq=hparam_yaml["train_freq"],
                exploration_final_eps=hparam_yaml["exploration_final_eps"],
                target_update_interval=hparam_yaml["target_update_interval"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    elif hparam_yaml["learner"] == "DDQN":
        model = DoubleDQN(hparam_yaml["policy_type"], env,
                learning_rate=hparam_yaml["lr"],
                batch_size=hparam_yaml["batch_size"],
                learning_starts=hparam_yaml["learning_starts"],
                train_freq=hparam_yaml["train_freq"],
                exploration_final_eps=hparam_yaml["exploration_final_eps"],
                target_update_interval=hparam_yaml["target_update_interval"],
                policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=f"runs/{run_id}")
    
    print(f"Training {hparam_yaml['learner']} on {hparam_yaml['env_name']} with {hparam_yaml['feat_dim']} features.")
    model.learn(total_timesteps=hparam_yaml["timesteps"])
    model.save(f"experiments/FeatAct_minigrid/models/{hparam_yaml['learner']}_{hparam_yaml['env_name']}")

    if hparam_yaml['analyse_rep']:
        # Analyse the agent's representation
        from analyse_rep import get_feats, see_feats
        feature_activations, _ = get_feats(model, hparam_yaml)
        print("got feat acts")
        see_feats(feature_activations)
        

if __name__ == '__main__':

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
        default='MiniGrid-Empty-5x5-v0', 
        help='Minigrid environment official name.'
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
        '--analyse_rep', 
        action='store_true', # set to false if we do not pass this argument
        help='Raise the flag to analyse feature vector.'
    )
    parser.add_argument(
        '--use_wandb', 
        action='store_true', # set to false if we do not pass this argument
        help='Raise the flag to use wandb.'
    )
    args = parser.parse_args()
    main(args)