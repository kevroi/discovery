from agents.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecVideoRecorder,
    VecTransposeImage,
)
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from stable_baselines3.common.env_util import make_vec_env
import minigrid
from minigrid.wrappers import (
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    FullyObsWrapper,
)
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from discovery.utils.feat_extractors import MinigridFeaturesExtractor
from discovery.experiments.feat_att_minigrid.n_room_env import (
    TwoRoomEnv,
    FourRoomEnv,
    FourRoomChainEnv,
    SixRoomChainEnv,
)

print("all loaded")

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1e5,
    # "env_name": "DiscoveryEnvs/MGFourRoomChainEnv-v0",
    "env_name": "MiniGrid-Empty-8x8-v0",
    "feat_dim": 128,
    "fully_obs": True,
    "random_starts": True,
    "random_goal": False,
    "goal_set": [(24, 6), (18, 6), (11, 6), (4, 6)],
    "n_envs": 1,
}

# Register your Gym Env
myEnv_id = "DiscoveryEnvs/MGFourRoomChainEnv-v0"  # It is best practice to have a space name and version number.
gym.envs.registration.register(
    id=myEnv_id,
    entry_point=FourRoomChainEnv,
    max_episode_steps=1e6,  # Customize to your needs.
    reward_threshold=1,  # Customize to your needs.
)

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


def make_env(config=config):
    # env = FourRoomChainEnv(render_mode="rgb_array",
    #                        random_starts=config["random_starts"],
    #                        random_goal=config["random_goal"])
    env = gym.make(config["env_name"], render_mode="rgb_array")
    if config["fully_obs"]:
        # env = FullyObsWrapper(env) # switch this to rgb for bigger scale exps
        env = RGBImgObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    # obs_space = env.observation_space
    # _, shapes, dtypes = obs_space_info(obs_space)
    # print(shapes)
    return env

    # # TODO Dont think these are 256 bit RGB image tensors

    # if config["fully_obs"]:
    #     image_wrappers = lambda env: ImgObsWrapper(RGBImgObsWrapper(env))
    # else:
    #     image_wrappers = lambda env: ImgObsWrapper(RGBImgPartialObsWrapper(env))

    # env = make_vec_env(env_id = config["env_name"],
    #                    n_envs=config["n_envs"],
    #                    monitor_dir=f"videos/{run.id}",
    #                    wrapper_class=image_wrappers,
    #                    env_kwargs=dict(render_mode="rgb_array",
    #                                    random_starts=config["random_starts"], # These are just for your custom envs
    #                                    random_goal=config["random_goal"],
    #                                     goal_set=config["goal_set"]
    #                                    )
    #                     )

    # return VecTransposeImage(env)


env = DummyVecEnv([make_env] * config["n_envs"])
# env = make_env()

env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 100000 == 0,
    video_length=200,
)

model = PPO(
    config["policy_type"],
    env,
    policy_kwargs=policy_kwargs,
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
