from gym.envs.registration import registry, register, make, spec
from kino_envs.env.differential_drive_env import DifferentialDriveEnv
from kino_envs.env.quadrotor_env import QuadrotorEnv

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

register(
    id='DifferentialDriveObsInv-v0',
    entry_point='kino_envs.env.differential_drive_env:DifferentialDriveObsEnvInv',
    max_episode_steps=DifferentialDriveEnv._max_episode_steps,
    reward_threshold=1.0,
)

register(
    id='DifferentialDriveObsAvoid-v0',
    entry_point='kino_envs.env.differential_drive_env:DifferentialDriveObsAvoidEnv',
    max_episode_steps=DifferentialDriveEnv._max_episode_steps,
    reward_threshold=1.0,
)
register(
    id='DifferentialDriveNoObs-v0',
    entry_point='kino_envs.env.differential_drive_env:DifferentialDriveNoObsEnv',
    max_episode_steps=DifferentialDriveEnv._max_episode_steps,
    reward_threshold=1.0,
)

register(
    id='QuadrotorEnvXF-v0',
    entry_point='kino_envs.env.quadrotor_env:QuadrotorEnvXF',
    max_episode_steps=QuadrotorEnv._max_episode_steps,
    reward_threshold=1.0,
)