from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from base_env import BaseEnv
from gym import spaces
import transforms3d.euler as euler
import itertools
from rigid import Obstacle, RectObs, Rigid, CircleRobot, RectRobot, Vehicle
from draw import plot_obs_list, plot_problem_definition, plot_robot
import os


def normalize_angle(angle):
    norm_angle = angle % (2 * np.pi)
    if norm_angle > np.pi:
        norm_angle -= 2 * np.pi
    return norm_angle


def T_transform2d(aTb, bP):
    """ Coordinate frame transformation in 2-D
    Args:
        aTb: transform with rotation, translation
        bP: non-holonomic coordinates in b
    Returns:
        non-holonomic coordinates in a
    """
    if len(bP.shape) == 1:  # vector
        bP_ = np.concatenate((bP, np.ones(1)))
        aP_ = aTb @ bP_
        aP = aP_[:2]
    elif bP.shape[1] == 2:
        bP = bP.T
        bP_ = np.vstack((bP, np.ones((1, bP.shape[1]))))
        aP_ = aTb @ bP_
        aP = aP_[:2, :].T
    return aP

class C(O):
step_dt = 1.0/10.0  # action step size

# physical constrain
max_v = 2.0
min_v = 0.0
max_acc = 4.0
min_acc = -4.0

max_phi = np.pi / 6.0
d = 0.6
# local map representation
free = 0
occupancy = 1

import pickle 
class DubinEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Args:
        """
        self.step_dt = step_dt

        self.max_v = max_v
        self.min_v = min_v
        self.max_phi = max_phi
        self.min_phi = -self.max_phi

        # vehicle shape
        self.rigid_robot = Vehicle(d, np.array(
            [[-0.4, -0.6], [1.6, 0.6]]), color="k")

        self.workspace_bounds = np.array(
            [[-20, 20], [-20, 20]], dtype=float
        )
        # x, y, theta
        self.state_bounds = np.array(
            [[-20, 20], [-20, 20], [-np.pi, np.pi], [min_v, max_v]], dtype=float
        )

        # the control space
        self.cbounds = np.array(
            [[min_acc, max_acc], [self.min_phi, self.max_phi]],
        )
        self.bias = (self.cbounds[:, 1] + self.cbounds[:, 0])/2
        self.scale = (self.cbounds[:, 1] - self.cbounds[:, 0])/2

        # action is normalized control
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.cbounds),))

        # obstacles
        self.obs_list = []

        self.state = np.zeros(len(self.state_bounds))
        self.goal = np.zeros(len(self.state))




        # observations include the local map and the current and goal states
        self._init_sample_positions()
        self.local_map_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.local_map_shape
        )

        self.observation_space = {
            'dynamics':spaces.Box(low=-np.inf, high=np.inf, shape=(5,)), 
            'map':spaces.Box(low=-np.inf, high=np.inf, shape=self.local_map_shape)
            }

        obstacles = pickle.load(open(os.path.dirname(__file__)+'/dubin_obstacles_list.pkl','rb'))[0]
        obstacles = [RectObs(**obs) for obs in obstacles]
        self.set_obs(obstacles)

        self.current_time = 0.0
        self.max_time = 50.0
        self.get_outline_points()

    def get_outline_points(self):
        outline = self.rigid_robot.outline()
        outline_points = np.zeros((0, 2))
        for i in range(len(outline)-1):
            points = np.linspace(start=outline[i], stop=outline[i+1],
                                 num=int(round(np.linalg.norm(outline[i+1]-outline[i])/0.1)), endpoint=False)
            outline_points = np.concatenate((outline_points, points), axis=0)
        self.outline_points = outline_points


    def state_dot(self, state, t, input_u):
        """ODE

        Arguments:
            state {array} -- current state
            t {float} -- dt
            input_u {array} -- control signal

        Returns:
            array -- next state
        """
        state_dot = np.zeros_like(state)
        state_dot[0] = state[3] * np.cos(state[2])
        state_dot[1] = state[3] * np.sin(state[2])
        state_dot[2] = state[3] / self.rigid_robot.d * np.tan(input_u[1])
        state_dot[3] = input_u[0]
        return state_dot

    def motion(self, state, input_u, duration):
        state = integrate.odeint(self.state_dot, state, [
                                 0, duration], args=(input_u,))[1]
        state[2] = normalize_angle(state[2])
        state = np.clip(state, self.state_bounds[:,0], self.state_bounds[:,1])
        return state

    def reach(self, state, goal):
        x1 = np.block([state[:2], np.cos(
            state[2]), np.sin(state[2])])
        x2 = np.block([goal[:2], np.cos(
            goal[2]), np.sin(goal[2])])
        return np.linalg.norm(x1-x2) <= 1.5

    def add_obs(self, obs):
        assert isinstance(obs, Obstacle), "Unsupported obstacle type"
        self.obs_list.append(obs)

    def set_obs(self, obs_list):
        self.obs_list = []
        for obs in obs_list:
            self.add_obs(obs)

    def get_clearance(self, state):
        dis = 100.0
        self.rigid_robot.pose = state[:3]
        if len(self.obs_list) > 0:
            for k, v in enumerate(self.obs_list):
                dis = min(dis, self.rigid_robot.dis2obs(v))
        return dis

    def valid_point_check(self, points):
        """
        Check collision for a batch of points
        points: array_like
        """
        collision = np.zeros(len(points), dtype=bool)
        for obs in self.obs_list:
            collision = np.logical_or(
                collision, obs.points_in_obstacle(points))
        return np.logical_not(collision)

    def valid_state_check(self, state):
        valid = super().valid_state_check(state)
        if not valid:
            return False
        wRb = euler.euler2mat(0, 0, state[2])[:2, :2]
        wTb = np.block([[wRb, state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        outline_points = T_transform2d(wTb, self.outline_points)
        for obs in self.obs_list:
            if np.any(obs.points_in_obstacle(outline_points)):
                return False
        return True

    def _init_sample_positions(self, left=-5, right=5, backward=-4, forward=10):
        """
        the robot can sense the local environment by sampling the local map.

        Returns
        -------
        sample_positions contain the positions need to be sampled in the body coordinate system.
        """
        local_map_size = np.array(
            [left, right, backward, forward])  # left, right, behind, ahead
        sample_reso = 0.3
        lr = np.linspace(local_map_size[0], local_map_size[1], int(
            round((local_map_size[1]-local_map_size[0])/sample_reso))+1)
        ba = np.linspace(local_map_size[3], local_map_size[2], int(
            round((local_map_size[1]-local_map_size[0])/sample_reso))+1)
        sample_positions = np.array(list(itertools.product(ba, lr)))
        self.local_map_size = local_map_size
        # (channel, width, height)
        self.local_map_shape = (1, len(ba), len(lr))
        self.sample_reso = sample_reso
        self.sample_positions = sample_positions
        self.sample_dis = np.linalg.norm(sample_positions, axis=1).reshape(self.local_map_shape)

    def sample_local_map(self):
        """Sampling points of local map, 
        with value representing occupancy (0 is non-free, 255 is free)
        """
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        # world sample position
        wPos = T_transform2d(wTb, self.sample_positions)
        local_map_ = self.valid_point_check(wPos)
        local_map = np.array(local_map_, dtype=np.float32)
        local_map[local_map_] = free
        local_map[np.logical_not(local_map_)] = occupancy 
        return local_map

    def normalize_u(self, u):
        """Normalize the actual control signals within the range [-1,1]
        """
        return (u-self.bias)/self.scale

    ################### gym interface ########################
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        u = action * self.scale + self.bias
        dt = self.step_dt
        self.state = self.motion(self.state, u, dt)
        self.current_time += dt

        # obs
        obs = self._obs()

        # info
        info = {'goal': False,
                'goal_dis': 0.0,
                'collision': False,
                'clearance': 0.0,
                }
        x1 = np.block([self.state[:2], np.cos(
            self.state[2]), np.sin(self.state[2])])
        x2 = np.block([self.goal[:2], np.cos(
            self.goal[2]), np.sin(self.goal[2])])
        info['goal_dis'] = np.linalg.norm(x1-x2)  # SE(2) distance
        if info['goal_dis'] <= 1.5:
            info['goal'] = True
        if np.any(obs['local_map']==occupancy):
            # print(self.sample_dis[obs['local_map']==occupancy])
            info['clearance'] = min(np.min(self.sample_dis[obs['local_map']==occupancy]), 4.0)
        else:
            info['clearance'] = 4.0
        info['collision'] = not self.valid_state_check(self.state)

        # reward
        # reward = -0.1*info['goal_dis']+0.1*np.tanh(action[0])-0.1*np.tanh(
        #     action[1])+20.0*info['goal']-50.0*info['collision']+0.04*info['clearance']
        reward = -0.1*info['goal_dis']+20.0*info['goal']-50.0*info['collision']

        # done
        done = info['goal'] or info['collision'] or self.current_time >= self.max_time

        return obs, reward, done, info

    def _obs(self):
        local_map = self.sample_local_map()
        local_map = local_map.reshape(self.local_map_shape)
        # wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        # wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
        #                 [np.zeros((1, 2)), 1]])
        # bTw = np.linalg.inv(wTb)
        # b_goal_pos = T_transform2d(bTw, self.goal[:2])
        # b_theta = normalize_angle(self.goal[2] - self.state[2])
        # b_goal = np.block([b_goal_pos, np.cos(b_theta), np.sin(b_theta)])

        # goal configuration in the robot coordinate frame, local map
        # obs = (local_map, b_goal)

        dynamic_obs = np.array([self.state[0]-self.goal[0], self.state[1]-self.goal[1], self.state[2], self.state[3], self.goal[2]])
        obs = {'dynamics': dynamic_obs, 'local_map': local_map}
        return obs

    def reset(self, low=5, high=12, obs_list_list= None):
        if obs_list_list is not None:
            ind_obs = np.random.randint(0, len(obs_list_list))
            self.set_obs(obs_list_list[ind_obs])
        # assert len(self.obs_list_list) > 0, 'No training environments'
        # if len(self.obs_list_list) > 0:
        #     ind_obs = np.random.randint(0, len(self.obs_list_list))
        #     self.set_obs(self.obs_list_list[ind_obs])
        min_clearance = 2
        # sample a random start goal configuration
        start = np.zeros(len(self.state_bounds))
        goal = np.zeros(len(self.state_bounds))
        while True:  # random sample start and goal configuration
            # sample a valid state
            start[:] = np.random.uniform(
                self.state_bounds[:, 0], self.state_bounds[:, 1])
            if self.get_clearance(start) <= min_clearance:
                continue

            # sample a valid goal
            for _ in range(5):
                r = np.random.uniform(low, high)
                theta = np.random.uniform(-np.pi, np.pi)
                goal[0] = np.clip(start[0] + r*np.cos(theta),
                                  *self.state_bounds[0, :])
                goal[1] = np.clip(start[1] + r*np.sin(theta),
                                  *self.state_bounds[1, :])
                goal[2] = np.random.uniform(-np.pi, np.pi)
                if self.get_clearance(goal) > min_clearance:
                    break

            if self.get_clearance(start) > min_clearance and self.get_clearance(goal) > min_clearance and low < np.linalg.norm(start[:2]-goal[:2]) < high:
                break
        self.state = start
        self.goal = goal

        self.current_time = 0
        obs = self._obs()
        return obs

    def render(self, mode='human', plot_localwindow=True, t = 0.1):
        if not hasattr(self, 'ax'):
            fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.xticks([])
            plt.yticks([])

        self.ax.cla()  # clear things

        plot_obs_list(self.ax, self.obs_list)
        if plot_localwindow:
            self.plot_localmap()
        plot_problem_definition(self.ax, self.obs_list,
                                self.rigid_robot, self.state, self.goal)
        plot_robot(self.ax, self.rigid_robot, self.state[:3])
        self.ax.axis([-22, 22, -22, 22])
        if t is not None:
            plt.pause(0.1)
        return None
    ####################### gym interface ####################

    def plot_localmap(self):
        wRb = euler.euler2mat(0, 0, self.state[2])[:2, :2]
        wTb = np.block([[wRb, self.state[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        tmp_wPos = T_transform2d(wTb, self.sample_positions)
        local_map = self.sample_local_map()
        ind_non_free = local_map == occupancy
        ind_free = local_map == free

        # plot boundary
        left_bot = [self.local_map_size[2], self.local_map_size[0]]
        left_top = [self.local_map_size[2], self.local_map_size[1]]
        rigit_top = [self.local_map_size[3], self.local_map_size[1]]
        rigit_bot = [self.local_map_size[3], self.local_map_size[0]]
        boundry = np.array([left_bot, left_top, rigit_top,
                            rigit_bot, left_bot], dtype=np.float32)
        boundry = T_transform2d(wTb, boundry)
        plt.plot(boundry[:, 0], boundry[:, 1], c='cyan')
        plt.plot(tmp_wPos[ind_free, 0], tmp_wPos[ind_free, 1],
                 '.', c='g', markersize=1)
        plt.plot(tmp_wPos[ind_non_free, 0],
                 tmp_wPos[ind_non_free, 1], '.', c='purple', markersize=1)

    def get_bounds(self):
        return {
            'workspace_bounds': self.workspace_bounds,
            'state_bounds': self.state_bounds,
            'cbounds': self.cbounds
        }

    def distance(self): pass


#################### collision-unaware version #################


class DubinEnvCU(DubinEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = self.state_space
        self.max_time = 50.0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = -0.1*info['goal_dis'] + 20.0*info['goal']
        return obs, reward, done, info

    def _obs(self):
        obs = super()._obs()['dynamics']  # without local map
        return obs



def visualize():
    env = DubinEnv()
    env.reset()
    env.render()
    done = False
    while not done:
        obs, reward, done, info = env.step(np.array([1.0,0.0]))
        print(obs)
        # print(info)
        env.render()
if __name__ == "__main__":
    visualize()
    # dubin_curve()
