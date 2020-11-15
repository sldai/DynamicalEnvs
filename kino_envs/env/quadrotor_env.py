import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/..')
from system.quadrotor import *
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import itertools
from gym import spaces
import gym

def valid_state(state, obs_list, width=1.0, radius=0.25):
    for obs in obs_list:
        corners = centered_box_to_points_3d(center=obs, size=[width]*3)
        obs_min_max = [np.min(corners, axis=0), np.max(corners, axis=0)]
        quadrotor_frame = rot_frame_3d(state, radius)   
        quadrotor_min_max = [np.min(quadrotor_frame, axis=1), np.max(quadrotor_frame, axis=1)]
        if quadrotor_min_max[0][0] <= obs_min_max[1][0] and quadrotor_min_max[1][0] >= obs_min_max[0][0] and\
            quadrotor_min_max[0][1] <= obs_min_max[1][1] and quadrotor_min_max[1][1] >= obs_min_max[0][1] and\
            quadrotor_min_max[0][2] <= obs_min_max[1][2] and quadrotor_min_max[1][2] >= obs_min_max[0][2]:
                return False
    return True

def point_distance_rectangle(p, rect):
    # function distance(rect, p) {
    # var dx = Math.max(rect.min.x - p.x, 0, p.x - rect.max.x);
    # var dy = Math.max(rect.min.y - p.y, 0, p.y - rect.max.y);
    # return Math.sqrt(dx*dx + dy*dy);
    # }
    pass

def points_in_obstacle(pos, size, points):
    c = np.abs(points - pos) <= size / 2
    return np.logical_and(np.logical_and(c[:, 0], c[:, 1]),c[:, 2])


class LocalMap(object):
    """A square local map used for sensing the enviroment 
    """
    occupancy_value = 1.0  # pixel value of the obstacle, the free space and the robot
    free_value = 0.0
    robot_value = -1.0
    def __init__(self, size=2.0, res=0.2, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.res = res
        x_min_max = np.linspace(-size/2,size/2,int(size/res)+1)
        y_min_max = np.linspace(-size/2,size/2,int(size/res)+1)
        z_min_max = np.linspace(-size/2,size/2,int(size/res)+1)
        self.shape = [1, len(x_min_max), len(y_min_max), len(z_min_max)]  # channel, x, y
        self.points = np.array(list(itertools.product(x_min_max, y_min_max, z_min_max)))

    def sample(self, obs_list, x, y, z):
        """Sample the enviroment to get the map
        """
        Wpoints = self.points + np.array([x,y,z])  # points in the world frame
        pixels = np.zeros(self.points.shape[0])
        pixels[:] = self.free_value
        for obs in obs_list:
            pixels[points_in_obstacle(obs, param.width, Wpoints)] = self.occupancy_value  # occupancy

        # reshape to the image shape
        Wpoints = Wpoints.reshape(self.shape+[3])
        pixels = pixels.reshape(self.shape)  # one channel
        return Wpoints.copy(), pixels.copy()

class QuadrotorEnv(Quadrotor, gym.Env):
    _max_episode_steps = int(30 / param.dt)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_list = obs_lists[0]
        self.local_map = LocalMap(res=0.2,size=2.0)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-10, 10, (13,)),
                "local_map": spaces.Box(-10,10,self.local_map.shape),
                "achieved_goal": spaces.Box(-10, 10, (2,)),
                "desired_goal": spaces.Box(-10, 10, (2,)),
            }
        )

        # action is normalized control signal
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.control_bias = (
            np.array([self.MAX_C1, self.MAX_C, self.MAX_C, self.MAX_C])
            + np.array([self.MIN_C1, self.MIN_C, self.MIN_C, self.MIN_C])
        ) / 2
        self.control_scale = (
            np.array([self.MAX_C1, self.MAX_C, self.MAX_C, self.MAX_C])
            - np.array([self.MIN_C1, self.MIN_C, self.MIN_C, self.MIN_C])
        ) / 2

        

    def distance(self,state1,state2):
        return np.linalg.norm(state1[:3]-state2[:3])

    def step(self, action):
        u = action.copy()
        u = u * self.control_scale + self.control_bias
        self.state = self.propagate(self.state.copy(), u.copy(), param.dt)
        self.cur_step += 1

        obs = self.get_obs()
        info = {
            "goal": self.distance(self.state, self.goal)<=param.goal_radius,
            "collision": not self.valid_state(self.state),
            "control": u,
            "current time": self.cur_step * param.dt,
            "is_success": self.distance(self.state, self.goal)<=param.goal_radius
        }
        reward = self.compute_reward(info)

        done = (
            info["goal"]
            or info["collision"]
            or self.cur_step >= self._max_episode_steps
        )
        return obs, reward, done, info

    def compute_reward(self, info):
        return (-12*info['collision'] + 5*info['goal'] - 0.01*self.distance(self.state, self.goal))*0.1

    def get_obs(self):
        points, pixels = self.local_map.sample(self.obs_list, *self.state[:3])
        return {
            'state': self.state,
            'local_map': pixels,
            'achieved_goal': self.state[:3],
            'desired_goal': self.goal[:3],
        }

    def reset(self):
        r = lambda :np.random.uniform(param.MIN_X, param.MAX_X)
        while True:
            start_state = np.array([r(), r(), r(),
                            0, 0, 0, 1,
                            0, 0, 0,
                            0, 0, 0])
            goal_state = np.array([r(), r(), r(),
                            0, 0, 0, 1,
                            0, 0, 0,
                            0, 0, 0])
            if not (valid_state(start_state, self.obs_list, width=param.width, radius=2) and valid_state(goal_state, self.obs_list, radius=2)) or np.linalg.norm(start_state-goal_state) < 3.0:
                continue
            else:
                break
        self.state = start_state
        self.goal = goal_state
        self.cur_step = 0
        return self.get_obs()

    def render(self, mode='human'):
        if not hasattr(self, 'ax'):
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
            self.ax = ax
        plt.cla()
        for obs in self.obs_list:
            draw_box_3d(self.ax, centered_box_to_points_3d(center=obs, size=[param.width]*3))
        draw_quadrotor_frame(self.ax, self.state, 'blue')
        draw_quadrotor_frame(self.ax, self.goal, 'red')
        points, pixels = self.local_map.sample(self.obs_list, *self.state[:3])
        
        occupancy = points[pixels==LocalMap.occupancy_value]
        if len(occupancy)>0:
            self.ax.plot(occupancy[:,0],occupancy[:,1],occupancy[:,2],'.r')
        self.ax.set_xlim3d(param.MIN_X, param.MAX_X)
        self.ax.set_ylim3d(param.MIN_X, param.MAX_X)
        self.ax.set_zlim3d(param.MIN_X, param.MAX_X)
        plt.pause(param.dt)

class QuadrotorEnvXF(QuadrotorEnv):
    def step(self, action):
        u = action.copy()
        u = u * self.control_scale + self.control_bias
        self.state = self.propagate(self.state.copy(), u.copy(), param.dt)
        self.cur_step += 1

        obs = self.get_obs()
        info = {
            "goal": self.distance(self.state, self.goal)<=param.goal_radius,
            "collision": not self.valid_state(self.state),
            "control": u,
            "current time": self.cur_step * param.dt,
            "is_success": self.distance(self.state, self.goal)<=param.goal_radius
        }
        reward = self.compute_reward(info)

        done = (
            info["collision"]
            or self.cur_step >= self._max_episode_steps
        )
        return obs, reward, done, info

    def compute_reward(self, info):
        velocity_term = -0.2*self.state[7] - 0.2*abs(self.state[8]) - 0.2*abs(self.state[9]) - 0.1*np.linalg.norm(self.state[10:13])
        b, c, d, a = self.state[3:7]
        rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                        [2 * b * c - 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d + 2 * a * b],
                        [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
        y_axis = rot_mat @ np.array([0,0,1.0])
        rew = -12*info['collision'] + (0.1*y_axis[2] + velocity_term)*0.1 - 0.1*self.distance(self.state, self.goal)
        return rew


if __name__ == "__main__":
    env = QuadrotorEnv()
    env.reset()
    env.render()
    for i in range(40):

        obs, rew, done, info = env.step([1.0, -1.0, 1.0, 0.0])
        print(obs, rew, done, info)
        env.render()
        if done:
            break
        
    plt.show()


