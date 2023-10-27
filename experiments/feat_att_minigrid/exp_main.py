import minigrid
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from cnn import MinigridFeaturesExtractor
import wandb
from wandb.integration.sb3 import WandbCallback
from two_room_env import TwoRoomEnv

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5e5,
    # "env_name": "MiniGrid-Empty-5x5-v0",
    "env_name": "Two-Rooms",
    "feat_dim":8,
    "fully_obs":True,
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

# def make_env(full_obs=config["fully_obs"]):
#     env = gym.make(config["env_name"], render_mode="rgb_array")
#     if full_obs:
#         env = FullyObsWrapper(env)
#     env = ImgObsWrapper(env)
#     env = Monitor(env)
#     return env

def make_env(full_obs=config["fully_obs"]):
    env = TwoRoomEnv(render_mode="rgb_array")
    if full_obs:
        env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(
                        env,
                        f"videos/{run.id}",
                        record_video_trigger=lambda x: x % 10000 == 0,
                        video_length=200,
                    )

model = PPO(config["policy_type"], env, policy_kwargs=policy_kwargs,
            verbose=1, tensorboard_log=f"runs/{run.id}")
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