import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/..")
from system.differential_drive import DifferentialDrive
from system.differential_drive import param, draw_base, base_points
from system.rigid import CircleObs, RectangleObs
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
import gym
import itertools
import pickle


def valid_state(state, base_points, obs_list):
    for obs in obs_list:
        if np.any(obs.points_in_obstacle(base_points.get_points_world_frame(*state[:3]))):
            return False
    return True

class DifferentialDriveEnv(DifferentialDrive, gym.GoalEnv):
    _max_episode_steps = int(30 / param.dt)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-20, 20, (5,)),
                "achieved_goal": spaces.Box(-20, 20, (2,)),
                "desired_goal": spaces.Box(-20, 20, (2,)),
            }
        )

        # action is normalized control signal
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.control_bias = (
            np.array([param.max_acc_v, param.max_acc_w])
            + np.array([-param.max_acc_v, -param.max_acc_w])
        ) / 2
        self.control_scale = (
            np.array([param.max_acc_v, param.max_acc_w])
            - np.array([-param.max_acc_v, -param.max_acc_w])
        ) / 2
        

    def step(self, action):
        u = action.copy()
        u = u * self.control_scale + self.control_bias
        self.state = self.propagate(self.state.copy(), u.copy(), param.dt)
        self.cur_step += 1

        obs = self._obs()
        info = {
            "goal": self.distance(self.state, self.goal)<=param.goal_radius,
            "collision": not self.valid_state(self.state),
            "control": u,
            "current time": self.cur_step * param.dt,
        }
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        done = (
            info["goal"]
            or info["collision"]
            or self.cur_step >= self._max_episode_steps
        )
        return obs, reward, done, info

    def _obs(self):
        obs = {
            "observation": np.array(
                [
                    self.state[0],
                    self.state[1],
                    self.state[2],
                    self.state[3],
                    self.state[4],
                ]
            ),
            "achieved_goal": self.state[:2],
            "desired_goal": self.goal[:2],
        }
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        """
        if info["collision"]:
            return -2.0
        elif info["goal"]:
            return 1.0
        else:
            return -0.1

    def reset(self):
        """
        """
        dis_range = [5.0, 12.0]
        start = np.zeros(5)
        goal = np.zeros(5)

        large_base_points = base_points(res=0.1, radius=param.v_r*2)  # used to sample states with clearance
        
        while True:
            start[:3] = np.random.uniform(
                np.array([param.x_min, param.y_min, -np.pi]),
                np.array([param.x_max, param.y_max, np.pi]),
            )
            goal[:3] = np.random.uniform(
                np.array([param.x_min, param.y_min, -np.pi]),
                np.array([param.x_max, param.y_max, np.pi]),
            )
            if not (valid_state(start, large_base_points, self.obs_list) and valid_state(goal, large_base_points, self.obs_list)):
                continue
            if dis_range[0] <= self.distance(start, goal) <= dis_range[1]:
                break
        self.state = start
        self.goal = goal

        self.cur_step = 0
        return self._obs()

    @staticmethod
    def distance(state1, state2):
        """Euclidean distance in 2D plane
        """
        return np.linalg.norm(state1[:2] - state2[:2])

    def render(self, mode='human', **kwargs):
        if not hasattr(self, 'ax'):
            fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.xticks([])
            plt.yticks([])

        self.ax.cla()  # clear things
        # draw obstacles
        for obs in self.obs_list:
            obs.draw(self.ax)

        # draw the robot
        draw_base(self.ax, self.state[0], self.state[1], self.state[2])
        # draw goal
        draw_base(self.ax, self.goal[0], self.goal[1], self.goal[2], color='red')

        self.ax.axis([param.x_min, param.x_max, param.y_min, param.y_max])
        plt.pause(param.dt)

class LocalMap(object):
    """A square local map used for sensing the enviroment 
    """
    occupancy_value = 1.0  # pixel value of the obstacle, the free space and the robot
    free_value = 0.0
    robot_value = -1.0
    def __init__(self, size=10.0, res=0.2, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.res = res
        x_min_max = np.linspace(-size/2,size/2,int(size/res)+1)
        y_min_max = np.linspace(-size/2,size/2,int(size/res)+1)
        self.shape = (len(x_min_max), len(y_min_max))
        self.points = np.array(list(itertools.product(x_min_max, y_min_max)))


    def sample(self, obs_list, x, y, yaw):
        """Sample the enviroment to get the map
        """
        Wpoints = self.points + np.array([x,y])  # points in the world frame
        pixels = np.zeros(self.points.shape[0])
        pixels[:] = self.free_value
        for obs in obs_list:
            pixels[obs.points_in_obstacle(Wpoints)] = self.occupancy_value  # occupancy
        pixels[np.linalg.norm(self.points, axis=1) <= param.v_r] = self.robot_value  # robot

        # reshape to the image shape
        Wpoints = Wpoints.reshape([self.shape[0],self.shape[1],2])
        pixels = pixels.reshape([self.shape[0],self.shape[1], 1])  # one channel
        return Wpoints.copy(), pixels.copy()
    
    def draw(self, ax, obs_list, x, y, yaw, obs_color='purple', robot_color='orange', free_color='cyan'):
        # draw local map
        points, pixels = self.sample(obs_list,x,y,yaw)
        points = points.reshape((np.prod(pixels.shape),2))
        pixels = pixels.reshape((np.prod(pixels.shape),))
        occ_points = points[pixels==self.occupancy_value]
        free_points = points[pixels==self.free_value]
        robot_points = points[pixels==self.robot_value]
        if len(free_points)>0:
            ax.plot(free_points[:,0], free_points[:,1], '.', color=free_color, markersize=0.5)
        if len(occ_points)>0:
            ax.plot(occ_points[:,0], occ_points[:,1], '.', color=obs_color, markersize=0.5)
        if len(robot_points)>0:
            ax.plot(robot_points[:,0], robot_points[:,1], '.', color=robot_color, markersize=0.5)




class DifferentialDriveObsEnv(DifferentialDriveEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        obs_path = f'{os.path.dirname(__file__)}/../../obstacles/differential_drive/0.pkl'
        self.set_obs_list([RectangleObs(**obs_param) for obs_param in pickle.load(open(obs_path,'rb'))])
        self.local_map = LocalMap(res=0.1)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-20, 20, (5,)),
                "local_map": spaces.Box(-1.0, 1.0, (self.local_map.shape[0], self.local_map.shape[1], 1)),
                "achieved_goal": spaces.Box(-20, 20, (2,)),
                "desired_goal": spaces.Box(-20, 20, (2,)),
            }
        )
    
    def _obs(self):
        obs = {
            "state": np.array(
                [
                    self.state[0],
                    self.state[1],
                    self.state[2],
                    self.state[3],
                    self.state[4],
                ]
            ),
            "local_map": self.local_map.sample(self.obs_list, self.state[0], self.state[1], self.state[2])[1],
            "achieved_goal": self.state[:2],
            "desired_goal": self.goal[:2],
        }
        return obs
    
    def render(self, mode='human', **kwargs):
        if not hasattr(self, 'ax'):
            fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.xticks([])
            plt.yticks([])

        self.ax.cla()  # clear things
        # draw obstacles
        for obs in self.obs_list:
            obs.draw(self.ax)
        if mode == 'local_map':
            # draw local map
            self.local_map.draw(self.ax, self.obs_list, *self.state[:3])
        else:
            # draw the robot
            draw_base(self.ax, self.state[0], self.state[1], self.state[2])
        # draw goal
        draw_base(self.ax, self.goal[0], self.goal[1], self.goal[2], color='red')
        


        self.ax.axis([param.x_min, param.x_max, param.y_min, param.y_max])
        plt.pause(param.dt)

