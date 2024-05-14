import numpy as np
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from discovery.utils.feat_extractors import MinigridFeaturesExtractor
import wandb
from wandb.integration.sb3 import WandbCallback
from discovery.environments.custom_minigrids import TwoRoomEnv

MAX_EPISODES = 1000
HALLWAY_LOCS = [(3, 7), (5, 7), (1, 7)]
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1000,
    "feat_dim": 12,
    "fully_obs": True,
}
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=config["feat_dim"]),
)
model = PPO(
    config["policy_type"],
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
)
# Do I save and load the model each time the environment changes?


for i in range(MAX_EPISODES):
    # randomly select hallway from HALLWAY_LOCS
    hallway_pos = HALLWAY_LOCS[np.random.randint(0, 3)]
    config["env_name"] = f"Two-Rooms-{hallway_pos}"
    config["hallway_pos"] = hallway_pos

    def make_tworoom_env(config=config):
        env = TwoRoomEnv(render_mode="rgb_array", hallway_pos=config["hallway_pos"])
        if config["fully_obs"]:
            env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env

    run = wandb.init(
        project="PPO_Rand_TwoRoom",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    env = DummyVecEnv([make_tworoom_env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 10000 == 0,
        video_length=200,
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
