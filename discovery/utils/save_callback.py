import os
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Saving new best model to {self.save_path}")
            self.model.save(self.save_path)

        return True

# class SaveModelCallback(BaseCallback):


# class SaveModel:

#     def __init__(
#         self,
#         model,
#         save_path: str,
#         name_prefix: str = "rl_model",
#         verbose: int = 0,
#     ):
#         self.model = model
#         self.save_path = save_path
#         self.name_prefix = name_prefix

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
#         """
#         Helper to get checkpoint path for each type of checkpoint.

#         :param checkpoint_type: empty for the model, "replay_buffer_"
#             or "vecnormalize_" for the other checkpoints.
#         :param extension: Checkpoint file extension (zip for model, pkl for others)
#         :return: Path to the checkpoint
#         """
#         return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

#     def __call__(self):
#         """Save the model"""

# class EveryNTimesteps(EventCallback):
#     """
#     Trigger a callback every ``n_steps`` timesteps

#     :param n_steps: Number of timesteps between two trigger.
#     :param callback: Callback that will be called
#         when the event is triggered.
#     """

#     def __init__(self, n_steps: int, callback: BaseCallback):
#         super().__init__(callback)
#         self.n_steps = n_steps
#         self.last_time_trigger = 0

#     def _on_step(self) -> bool:
#         if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
#             self.last_time_trigger = self.num_timesteps
#             return self._on_event()
#         return True
