from gym.envs.registration import registry, register, make, spec
from kino_envs.env.differential_drive_env import DifferentialDriveEnv

# Algorithmic
# ----------------------------------------

register(
    id='DifferentialDrive-v0',
    entry_point='kino_envs.env.differential_drive_env:DifferentialDriveEnv',
    max_episode_steps=DifferentialDriveEnv._max_episode_steps,
    reward_threshold=1.0,
)

register(
    id='DifferentialDriveObs-v0',
    entry_point='kino_envs.env.differential_drive_env:DifferentialDriveObsEnv',
    max_episode_steps=DifferentialDriveEnv._max_episode_steps,
    reward_threshold=1.0,
)