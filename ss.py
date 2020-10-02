import gym
import kino_envs
import numpy as np
import matplotlib.pyplot as plt
from kino_envs.system.rigid import *
from kino_envs.env.differential_drive_env import DifferentialDriveObsEnv, DifferentialDriveEnv
import pickle

if __name__ == "__main__":
    env = gym.make('DifferentialDriveObs-v0')
    # env.set_obs_list([RectangleObs(**obs_param) for obs_param in pickle.load(open('obstacles/dubin_obstacles/dubin_obstacles_0.pkl','rb'))])
    # print(env.env.obs_list)
    env.reset()
    env.render()
    while True:
        obs, r, done, info = env.step(np.array([1.,0]))
        # print(env.state, env.goal)
        # print(obs, r, done, info)
        print(info)
        env.render(mode='local_map')
        if done:
            break
    plt.show()