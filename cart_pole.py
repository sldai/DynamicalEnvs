
from scipy import integrate
from rl_planner.env.base_env import BaseEnv
import numpy as np
from numpy import cos, sin
from rl_planner.env.dubin import normalize_angle
from rl_planner.utils.draw import plot_obs_list
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rl_planner.env.rigid import CartPole, Obstacle
import gym
import gym.spaces as spaces

I = 10
L = 2.5
M = 10
m = 5
g = 9.8

STATE_X = 0
STATE_V = 1
STATE_THETA = 2
STATE_W = 3
CONTROL_A = 0

MIN_X = -30
MAX_X = 30
MIN_V = -40
MAX_V = 40
MIN_W = -4
MAX_W = 4

MAX_F = 300


class CartPoleEnv(BaseEnv):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             MIN_X                   MAX_X
        1       Cart Velocity             MIN_V                   MAX_V
        2       Pole Angle                -pi                     pi
        3       Pole Angular Velocity     MIN_W                   MAX_W
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_dt = 0.05
        self.step_dt = 0.01

        self.state_bounds = np.array(
            [
                [MIN_X, MAX_X],
                [MIN_V, MAX_V],
                [-np.pi, np.pi],
                [MIN_W, MAX_W]
            ]
        )
        self.cbounds = np.array(
            [[-MAX_F, MAX_F]], dtype=float
        )
        self.state = np.zeros(len(self.state_bounds))
        self.goal = self.state.copy()
        self.action_space = spaces.Box(
            low=self.cbounds[:, 0], high=self.cbounds[:, 1])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,))

        self.rigid_robot = CartPole(rect=np.array(
            [[-1.0, -1.0], [1.0, 1.0]]), l=2*L, x=0.0, theta=0.0)
        self.obs_list = []

        self.current_time = 0.0
        self.max_time = 3.0

    def deriv(self, state, t, input_u):
        _x, _v, _theta, _w = state
        _a = input_u[0]
        mass_term = (M + m)*(I + m * L * L) - m * m * \
                     L * L * cos(_theta) * cos(_theta)

        deriv = np.zeros_like(state)
        deriv[0] = _v
        deriv[2] = _w
        mass_term = (1.0 / mass_term)
        deriv[1] = ((I + m * L * L)*(_a + m * L * _w * _w * sin(_theta)) +
                    m * m * L * L * cos(_theta) * sin(_theta) * g) * mass_term
        deriv[3] = ((-m * L * cos(_theta))*(_a + m * L * _w * _w *
                    sin(_theta))+(M + m)*(-m * g * L * sin(_theta))) * mass_term
        return deriv

    def motion(self, state, input_u, duration):
        """ODE for the cartpole
        .. seealso: https://arxiv.org/pdf/1405.2872.pdf
        """
        state = state.copy()
        state = integrate.odeint(self.deriv, state, [0, duration], args=(input_u,))[-1]

        state[2] = normalize_angle(state[2])
        for i in range(len(state)):
            state[i] = np.clip(state[i],*self.state_bounds[i,:])
        return state

    def valid_state_check(self, state):
        valid = super().valid_state_check(state)
        if not valid:
            return False
        self.rigid_robot.set_config(state[0], state[2])
        for obs in self.obs_list:
            if self.rigid_robot.collision_obs(obs):
                return False
        return True

    def reach(self, state, goal):
        return bool(abs(state[0]-goal[0])<=2.0 
                    and abs(normalize_angle(state[2]-goal[2]))<=30*np.pi/180)
        # return self.distance(state, goal)
    

    @staticmethod
    def distance(x1, x2):
        return (((x1[0]-x2[0]))**2+(3*(cos(x1[2])-cos(x2[2])))**2+(3*(sin(x1[2])-sin(x2[2])))**2)**0.5

    def step(self, action):
        dt = self.step_dt
        self.state = self.motion(self.state, action, dt)
        self.current_time += dt
  
        # obs
        obs = self._obs()

        # info
        info = {'goal': False,
                'collision': False,
                }
        info['goal'] = self.reach(self.state, self.goal)
        info['collision'] = not self.valid_state_check(self.state)

        # reward
        reward = -0.05*self.distance(self.state, self.goal) + 20.0*info['goal'] -0.1

        # done
        done = info['goal'] or info['collision'] or self.current_time >= self.max_time

        return obs, reward, done, info

    def _obs(self):
        obs = np.zeros(5)
        obs[:3] = self.state[1:]
        obs[3] = self.goal[0] - self.state[0]
        obs[4] = self.goal[2]
        return obs

    
    def reset(self, low=2.0, high=6.0):
        while True:
            start = np.random.uniform(low=self.state_bounds[:, 0], high=self.state_bounds[:,1])
            goal = np.random.uniform(low=self.state_bounds[:, 0], high=self.state_bounds[:,1])
            if self.valid_state_check(start) and self.valid_state_check(goal) and low<self.distance(start,goal)<high:
                break
        self.state = start
        self.goal = goal
        self.current_time = 0.0
        return self._obs()

    def render(self):
        if not hasattr(self, 'ax'):
            fig, self.ax = plt.subplots()
            plt.xticks([])
            plt.yticks([])

        self.ax.cla()  # clear things
        plt.plot(np.array([MIN_X, MAX_X]),np.array([0,0]),'-k')
        plt.axis('equal')

        plot_obs_list(self.ax, self.obs_list)
        width = self.rigid_robot.cart.rect[1, 0]-self.rigid_robot.cart.rect[0,0]
        height = self.rigid_robot.cart.rect[1, 1]-self.rigid_robot.cart.rect[0,1]
        cart = patches.Rectangle((self.state[0]-width/2, 0.0-height/2),
                                 width, height, color=self.rigid_robot.cart.color)
        self.ax.add_patch(cart)

        self.rigid_robot.set_config(self.state[0], self.state[2])
        pole = self.rigid_robot.get_pole()
        self.ax.plot(pole[:, 0], pole[:,1], color=self.rigid_robot.pole_color)

        self.rigid_robot.set_config(self.goal[0], self.goal[2])
        goal_pole = self.rigid_robot.get_pole()
        self.ax.plot(goal_pole[:, 0], goal_pole[:,1], color='cyan')

        plt.pause(0.1)
        return None

    def add_obs(self, obs):
        assert isinstance(obs, Obstacle), "Unsupported obstacle type"
        self.obs_list.append(obs)

    def set_obs(self, obs_list):
        self.obs_list = []
        for obs in obs_list:
            self.add_obs(obs)

# if __name__ == "__main__":
#     env = CartPoleEnv()
#     obs = env.reset()
#     env.render()

#     while True:
#         obs, reward, done, info = env.step(np.array([-300.0]))
#         print(env.state)
#         env.render()
#         if done:
#             break
