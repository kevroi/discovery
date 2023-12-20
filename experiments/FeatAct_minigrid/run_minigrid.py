from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from cnn import MinigridFeaturesExtractor
import wandb

## HELPER FUNCTIONS ##
def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env
######################

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1e5,
    "env_name": "MiniGrid-DoorKey-8x8-v0",
    "feat_dim":24,
    "record_video":True,
    "n_envs":1,
}

policy_kwargs = dict(
                    features_extractor_class=MinigridFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=config["feat_dim"]),
                    )

run = wandb.init(
                project="PPO_on_MiniGrid",
                config=config,
                sync_tensorboard=True, 
                monitor_gym=True,
                save_code=True,
                )

env = DummyVecEnv([make_env]*config["n_envs"])
if config["record_video"]:
    env = VecVideoRecorder(
                            env,
                            f"videos/{run.id}",
                            record_video_trigger=lambda x: x % 100000 == 0,
                            video_length=200,
                        )

model = PPO(config["policy_type"], env, policy_kwargs=policy_kwargs,
            verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=config["total_timesteps"],
            )
model.save(f'experiments/FeatAct_minigrid/models/ppo_{config["env_name"]}')