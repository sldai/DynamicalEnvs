import pickle
import gym
import imageio
import matplotlib.pyplot as plt
import kino_envs
from kino_envs.env.differential_drive_env import DifferentialDriveObsEnvInv

for_traj = pickle.load(open('forward.pkl','rb'))
env = DifferentialDriveObsEnvInv()
env.reset()
env.goal = for_traj[0][0].copy()
back_traj = []
for state, action, state_next, goal in for_traj[::-1]:
    state = state_next.copy()
    env.state = state_next.copy()
    # print(action)
    env.step(action)
    state_next = env.state.copy()
    print(state_next-state)
    back_traj.append([state, action, state_next, env.goal.copy()])
# images = []

# for state, action, state_next, goal in back_traj:
#     print(state_next)
#     env.goal = goal.copy()
#     env.state = state_next.copy()
#     env.render()
#     plt.savefig("examples/tmp.png")
#     images.append(plt.imread("examples/tmp.png"))
# imageio.mimsave("examples/xxx.gif", images, duration=0.1)
import numpy as np
for_path = np.array([state for state, action, state_next, goal in for_traj])
back_path = np.array([state_next for state, action, state_next, goal in back_traj])
back_path = back_path[::-1]
plt.figure()
plt.plot(for_path[:,0], for_path[:,1], label='forward')
plt.plot(back_path[:,0], back_path[:,1], label='backward')
print(back_path[:,3])
plt.legend()
plt.show()