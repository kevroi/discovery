print("loading")
from stable_baselines3.common.vec_env import VecVideoRecorder
print("loaded recorder")
from stable_baselines3.common.env_util import make_atari_env
print("loaded atari env")
from stable_baselines3.common.vec_env import VecFrameStack
print("loaded vec stack")
from stable_baselines3 import PPO
print("loaded ppo")

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
    "timesteps": 1e4,
    "record_video": False,
}

# run = wandb.init(
#                 project="PPO_on_Atari_Tests",
#                 config=config,
#                 sync_tensorboard=True, 
#                 # monitor_gym=True,
#                 save_code=True,
#                 )

env = make_env()
print("env made")
# if config["record_video"]:
#     env = VecVideoRecorder(
#                             env,
#                             f"videos/{run.id}",
#                             record_video_trigger=lambda x: x % 100000 == 0,
#                             video_length=200,
#                         )
    
model = PPO(config["policy_type"], env, verbose=1)
print("agent made")
model.learn(total_timesteps=config["timesteps"],
            )
print("learning done")

