from agents.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from stable_baselines3.common.env_util import make_vec_env
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper, FullyObsWrapper
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from cnn import MinigridFeaturesExtractor
print("all loaded")

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1e5,
    # "env_name": "DiscoveryEnvs/MGFourRoomChainEnv-v0",
    "env_name": "MiniGrid-DoorKey-5x5-v0",
    "feat_dim":12,
    "fully_obs":True,
    "random_starts":False,
    "random_goal":False,
    "goal_set": [],
    "n_envs":1,
}

run = wandb.init(
                project="PPO_on_MiniGrid",
                config=config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
                )

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=config["feat_dim"]),
)

def make_env(config=config):
    env = gym.make(config["env_name"], render_mode="rgb_array")
    if config["fully_obs"]:
        # env = FullyObsWrapper(env) # switch this to rgb for bigger scale exps
        env = RGBImgObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env]*config["n_envs"])
# env = make_env()

env = VecVideoRecorder(
                        env,
                        f"videos/{run.id}",
                        record_video_trigger=lambda x: x % 100000 == 0,
                        video_length=200,
                    )

model = PPO(config["policy_type"], env, policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            )
model.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
                ),
            )
run.finish()


# To test with no logging
# policy_kwargs = dict(
#     features_extractor_class=MinigridFeaturesExtractor,
#     features_extractor_kwargs=dict(features_dim=128),
# )

# env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
# env = ImgObsWrapper(env)

# env = TwoRoomEnv(render_mode="rgb_array")
# env = ImgObsWrapper(env)

# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(2e5)