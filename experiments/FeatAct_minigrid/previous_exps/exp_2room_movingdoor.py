import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from utils.cnn import MinigridFeaturesExtractor
import wandb
from wandb.integration.sb3 import WandbCallback
from experiments.feat_att_minigrid.n_room_env import TwoRoomEnv


#hypothesis - does the agent (tf5hze1v) retain the same feature activations once the door moves?
hallway_locs = [(4,7), (5,7), (1,7)] 

for hallway_pos in hallway_locs:
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 2.5e5,
        # "env_name": "MiniGrid-Empty-5x5-v0",
        "env_name": "Two-Rooms",
        "feat_dim":8,
        "fully_obs":True,
        "hallway_pos": hallway_pos
    }

    def make_tworoom_env(config=config):
        env = TwoRoomEnv(render_mode="rgb_array", hallway_pos=config["hallway_pos"])
        if config["fully_obs"]:
            env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env
    
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=config["feat_dim"]),
    )

    run = wandb.init(
                    project="PPO_on_MiniGrid",
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

    # load the agent that was trained in default two rooms, for 500k steps
    model = PPO.load("/Users/kevinroice/Documents/research/discovery/models/tf5hze1v/model",
                     env=env)

    model.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
                ),
            )
    run.finish()