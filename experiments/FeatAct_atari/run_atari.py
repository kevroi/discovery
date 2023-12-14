from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import wandb

## HELPER FUNCTIONS ##
def make_env():
    env = make_atari_env(config["env_name"],
                         n_envs=config["n_envs"],
                         seed=config["env_seed"])
    env = VecFrameStack(env, n_stack=4)
    return env
######################

config = {
    "env_name": "ALE/MontezumaRevenge-v5",
    "n_envs": 4, # TODO check if this means the same as PPO n_envs
    "env_seed": 0,
    "policy_type": "CnnPolicy",
    "timesteps": 1e5,
    "record_video": True,
}

run = wandb.init(
                project="PPO_on_Atari_Tests",
                config=config,
                sync_tensorboard=True, 
                monitor_gym=True,
                save_code=True,
                )

env = make_env()
if config["record_video"]:
    env = VecVideoRecorder(
                            env,
                            f"videos/{run.id}",
                            record_video_trigger=lambda x: x % 100000 == 0,
                            video_length=200,
                        )
    
model = PPO(config["policy_type"], env, verbose=1,
            tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=config["total_timesteps"],
            )

