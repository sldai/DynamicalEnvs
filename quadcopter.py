from scipy.linalg import solve_continuous_are, solve_discrete_are
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import transforms3d
from base_env import BaseEnv
import numpy as np
from gym import spaces
import itertools
from scipy import integrate


############################### quadcopter parameters #############################
mass = 0.2  # kg
g = 9.8  # m/s/s
Ix, Iy, Iz = 0.00025, 0.00025, 0.0003738
I = np.array([(0.00025, 0, 0),
              (0, 0.00025, 0),
              (0, 0, 0.0003738)])

invI = np.linalg.inv(I)

prop_radius = 0.04
arm_length = 0.086  # meter
height = 0.05
minF = 0.0
maxF = 2.0 * mass * g
L = arm_length
H = height
km = 1.5e-9
kf = 6.11e-8
r = km / kf  # drag coefficient

#  [ F  ]         [ F1 ]
#  | M1 |  = A *  | F2 |
#  | M2 |         | F3 |
#  [ M3 ]         [ F4 ]
A = np.array([[1,  1,  1,  1],
              [0,  L,  0, -L],
              [-L,  0,  L,  0],
              [r, -r,  r, -r]])

invA = np.linalg.inv(A)

body_frame = np.array([(L, 0, 0, 1),
                       (0, L, 0, 1),
                       (-L, 0, 0, 1),
                       (0, -L, 0, 1),
                       (0, 0, 0, 1),
                       (0, 0, H, 1)])


MIN_X = -1
MAX_X = 1
MIN_Y = -1
MAX_Y = 1
MIN_Z = 0
MAX_Z = 2

MIN_X_DOT = -1
MAX_X_DOT = 1
MIN_Y_DOT = -1
MAX_Y_DOT = 1
MIN_Z_DOT = -1
MAX_Z_DOT = 1

MIN_ROLL = -np.pi/10   # Force the quadcopter stay near the stable
MAX_ROLL = np.pi/10    # position, otherwise the simulation becomes inaccurate
MIN_PITCH = -np.pi/10
MAX_PITCH = np.pi/10
MIN_YAW = -np.pi
MAX_YAW = np.pi

MIN_ROLL_DOT = -np.pi*2
MAX_ROLL_DOT = np.pi*2
MIN_PITCH_DOT = -np.pi*2
MAX_PITCH_DOT = np.pi*2
MIN_YAW_DOT = -np.pi*2
MAX_YAW_DOT = np.pi*2

MIN_THRUST = minF
MAX_THRUST = maxF
MIN_ROLL_TORQUE = -L * maxF/4
MAX_ROLL_TORQUE = L * maxF/4
MIN_PITCH_TORQUE = -L * maxF/4
MAX_PITCH_TORQUE = L * maxF/4
MIN_YAW_TORQUE = -2 * r * maxF/4
MAX_YAW_TORQUE = 2 * r * maxF/4


def wrap_angle(angle):
    wrapped_angle = angle % (2 * np.pi)
    if wrapped_angle > np.pi:
        wrapped_angle -= 2 * np.pi
    return wrapped_angle


class QuadcopterEnv(BaseEnv):
    """ Quadcopter class

    state  - 1 dimensional vector but used as 13 x 1. [x, y, z, x_dot, y_dot, z_dot, roll, pitch, yaw, p, q, r]
    where [p, q, r] are angular velocity in the "body frame" 
    F      - 1 x 1, thrust output from controller
    M      - 3 x 1, moments output from controller
    params - system parameters struct, arm_length, g, mass, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_bounds = np.array([
            [MIN_X, MAX_X],  # x
            [MIN_Y, MAX_Y],  # y
            [MIN_Z, MAX_Z],  # z
            [MIN_X_DOT, MAX_X_DOT],  # xd
            [MIN_Y_DOT, MAX_Y_DOT],  # yd
            [MIN_Z_DOT, MAX_Z_DOT],  # zd
            [MIN_ROLL, MAX_ROLL],  # qw
            [MIN_PITCH, MAX_PITCH],  # qx
            [MIN_YAW, MAX_YAW],  # qy
            [MIN_ROLL_DOT, MAX_ROLL_DOT],  # p
            [MIN_PITCH_DOT, MAX_PITCH_DOT],  # q
            [MIN_YAW_DOT, MAX_YAW_DOT],  # r
        ], dtype=float)

        self.observation_space = spaces.Box(
            low=self.state_bounds[:, 0], high=self.state_bounds[:, 1],
        )

        self.cbounds = np.array(
            [[MIN_THRUST, MAX_THRUST],
             [MIN_ROLL_TORQUE, MAX_ROLL_TORQUE],
             [MIN_PITCH_TORQUE, MAX_PITCH_TORQUE],
             [MIN_YAW_TORQUE, MAX_YAW_TORQUE]]
        )

        # actions are normalized for the reinforcement learning
        self.bias = (self.cbounds[:, 1] + self.cbounds[:, 0])/2
        self.scale = (self.cbounds[:, 1] - self.cbounds[:, 0])/2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.cbounds),))

        self.dt = 0.1
        self.max_time = 10.0
        self.current_time = 0.0

        self.state = np.zeros_like(self.state_bounds[:, 0])
        self.goal = np.zeros_like(self.state_bounds[:, 0])

    def position(self):
        return self.state[0:3]

    def velocity(self):
        return self.state[3:6]

    def orientation(self):
        return self.state[6:9]

    def omega(self):
        return self.state[9:12]

    def state_dot(self, state, t, u):
        F, M = u[0], u[1:4]
        state_dot = np.zeros(12)
        # velocities
        state_dot[:3] = state[3:6]

        # accelerations
        wRb = transforms3d.euler.euler2mat(*state[6:9])
        state_dot[3:6] = 1.0 / mass * \
            (wRb @ np.array([0, 0, F]) - np.array([0, 0, mass * g]))

        # angular velocity
        # note that there is a assumption that the quadcopter is at the stable state,
        # i.e. the roll and pitch angle are close to 0. So the angular velocity in the body
        # frame can be directly used as the angular velocity in the inertial frame
        # see also: http://www.uta.edu/utari/acs/ee5323/notes/quadrotor%20Dynamic%20Inversion%20IET%20CTA%20published%20version.pdf
        state_dot[6:9] = state[9:12]

        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = state[9:12]
        omega_dot = invI @ (M - np.cross(omega, I @ omega))
        state_dot[9:12] = omega_dot
        return state_dot

    def motion(self, state, u, dt):
        """Propagate the dynamic
        """
        # limit the thrust and torques within the propeller
        prop_thrusts = invA @ u
        prop_thrusts_clamped = np.clip(prop_thrusts, minF/4, maxF/4)
        u = A @ prop_thrusts_clamped
        state = integrate.odeint(self.state_dot, state, [0, dt], args=(u,))[1]

        # wrap angle
        for i in range(6, 9):
            state[i] = wrap_angle(state[i])

        for i in range(len(state)):
            state[i] = np.clip(state[i], *self.state_bounds[i, :])
        return state

    def normalize_u(self, u):
        """Normalize the actual control signals within the range [-1,1]
        """
        return (u-self.bias)/self.scale

    def step(self, action):
        # action is normalized trust of each propepler
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # total thrust and Moment
        u = action * self.scale + self.bias
        self.state = self.motion(self.state, u, self.dt)
        self.current_time += self.dt

        # obs
        obs = self._obs()

        # info
        info = {
            'goal': self.reach(self.state, self.goal),
            'goal_dis': self.distance(self.state, self.goal),
            'collision': not self.valid_state_check(self.state)
        }

        # reward
        positionRewardCoeff_ = -1e-1
        thrustRewardCoeff_ = -1e-2
        orientationRewardCoeff_ = -1e-2*3
        angleVelRewardCoeff_ = -1e-2

        # penalize the distance to goal
        positionReward_ = positionRewardCoeff_ * \
            self.distance(self.state, self.goal)
        # penalize large torques
        thrustReward_ = thrustRewardCoeff_ * np.linalg.norm(action[1:])
        # penalize unstable orientations (roll and pitch are not 0)
        orientationReward_ = orientationRewardCoeff_ * \
            np.linalg.norm(self.orientation()[:2])
        # penalize large angular velocities
        angleVelReward_ = angleVelRewardCoeff_ * np.linalg.norm(self.omega())

        reward = 0.3+positionReward_+thrustReward_+orientationReward_ + \
            angleVelReward_+20.0*info['goal']-50.0*info['collision']
        # reward = -0.2*np.linalg.norm(self.state[3:5])+0.1*self.state[5]+thrustReward_+orientationReward_+angleVelReward_

        # done
        done = info['goal'] or info['collision'] or self.current_time >= self.max_time

        return obs, reward, done, info

    def _obs(self):
        obs = self.state.copy()
        obs[:3] -= self.goal[:3]
        return obs

    def distance(self, state, goal):
        return np.linalg.norm(state[:3]-goal[:3])

    def reach(self, state, goal):
        return self.distance(state, goal) <= 0.1

    def reset(self, low=0.2, high=0.5):
        start = np.zeros_like(self.state_bounds[:, 0])
        goal = np.zeros_like(self.state_bounds[:, 0])

        while True:
            start[:] = np.random.uniform(
                low=self.state_bounds[:, 0], high=self.state_bounds[:, 1])
            # start[2] = np.random.uniform(0.4, 2)
            # start[3:6] = 0

            goal[:3] = np.random.uniform(
                low=self.state_bounds[:3, 0], high=self.state_bounds[:3, 1])
            # goal[2] = np.random.uniform(0.4, 2)
            if self.valid_state_check(start) and self.valid_state_check(goal) and low <= self.distance(start, goal) <= high:
                break

        self.state = start
        self.goal = goal
        self.current_time = 0.0
        return self._obs()

    def world_frame(self):
        """ position returns a 3x6 matrix
            where row is [x, y, z] column is m1 m2 m3 m4 origin h
        """
        origin = self.state[0:3]
        rot = transforms3d.euler.euler2mat(*self.orientation())
        wHb = np.block(
            [[rot, origin.reshape((-1, 1))],
             [np.array([[0, 0, 0, 1]])]]
        )

        quadBodyFrame = body_frame.T
        quadWorldFrame = wHb.dot(quadBodyFrame)
        world_frame = quadWorldFrame[0:3]
        return world_frame

    def render(self, t=0.001):
        if not hasattr(self, 'ax'):
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            self.ax = ax

        self.ax.cla()
        # self.dynamic_model.state = self.euler2quad(self.state)
        frame = self.world_frame()
        self.ax.plot(frame[0, [0, 2]], frame[1, [0, 2]],
                     frame[2, [0, 2]], '-', marker='.', c='cyan', markeredgecolor='k', markerfacecolor='k')[0]
        self.ax.plot(frame[0, [1, 3]], frame[1, [1, 3]],
                     frame[2, [1, 3]], '-', marker='.', c='red', markeredgecolor='k', markerfacecolor='k')[0]
        self.ax.plot(frame[0, [4, 5]], frame[1, [4, 5]], frame[2, [
                     4, 5]], '-', c='blue')[0]

        self.ax.plot([self.goal[0]], [self.goal[1]],
                     [self.goal[2]], '-o', c='red')
        self.ax.set_xlim([MIN_X, MAX_X])
        self.ax.set_ylim([MIN_Y, MAX_Y])
        self.ax.set_zlim([MIN_Z, MAX_Z])
        plt.tight_layout()
        if t != None:
            plt.pause(t)

    def valid_state_check(self, state):
        valid = super().valid_state_check(state)
        # valid = valid and state[2] > self.state_bounds[2,0]
        return valid


class QuadcopterEnvV2(QuadcopterEnv):
    """Obstacles are added in this environment, the quadcopter learns to avoid collision and reach the goal region. It can sense surrounding obstacles and represent them in a 3-D local map.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the quadcopter body is approximated as a cuboid for collision checking
        self.geo_shape = np.array([
            [-arm_length-prop_radius,
                arm_length+prop_radius],
            [-arm_length-prop_radius,
                arm_length+prop_radius],
            [0, height]])
        sample_positions = []
        reso = 0.05
        for i in range(len(self.geo_shape)):
            sample_positions.append(
                np.linspace(self.geo_shape[i, 0],
                            self.geo_shape[i, 1],
                            int(np.round(
                                (self.geo_shape[i, 1]-self.geo_shape[i, 0])/reso)+1)
                            )
            )
        self.body_points = np.array(list(itertools.product(*sample_positions)))

        # TODO: add obstacles
        self.obstacles = []

        self.local_map_shape, self.local_map_points = self._init_local_map()
        self.observation_space = {
            'basic': spaces.Box(
                low=self.state_bounds[:, 0],
                high=self.state_bounds[:, 1]
            ),
            'local_map': spaces.Box(low=0, high=1, shape=self.local_map_shape)
        }

    def _init_local_map(self):
        """
        the robot can sense the local environment by sampling the local map.

        Returns
        -------
        local_map_shape: the local map shape, used for CNN 
        sample_positions: the positions need to be sampled in the body coordinate system.
        """
        size = 1.0
        reso = 0.2
        samples_x = np.linspace(-size, size, int(2*size/reso)+1)
        sample_positions = np.array(
            list(itertools.product(samples_x, samples_x, samples_x)))

        # (channel, length, width, height)
        local_map_shape = (1, len(samples_x), len(samples_x), len(samples_x))
        return local_map_shape, sample_positions

    def sample_local_map(self):
        """Sampling points of local map, 
        with value representing occupancy (0 is non-free, 1 is free)
        """
        origin = self.state[0:3]
        rot = transforms3d.quaternions.quat2mat(self.state[6:10])
        wTb = np.block(
            [[rot, origin.reshape((-1, 1))],
             [np.zeros((1, 3)), 1]]
        )
        # world sample position
        wPos = (wTb @ np.concatenate((self.local_map_points, np.ones(
            (len(self.local_map_points), 1))), axis=1).T).T[:, :3]
        local_map = self.valid_point_check(wPos).astype(np.float32)
        return local_map

    def valid_point_check(self, points):
        """Check collision for a batch of points
        Args:
            points (2d array): each row [x,y,z]

        Returns:
            valid (1d array): the element indicates 
            whether the corresponding point is valid
        """
        collision = np.zeros(len(points), dtype=bool)
        for obs in self.obstacles:
            collision = np.logical_or(
                collision, obs.points_in_obstacle(points))
        valid = np.logical_not(collision)
        return valid

    def valid_state_check(self, state):
        valid = super().valid_state_check(state)
        valid = valid and np.all(self.valid_point_check(self.body_points))
        return valid

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # TODO: modefy the reward function for collsion avoidance
        return obs, reward, done, info

    def render(self, t=0.001):
        if not hasattr(self, 'ax'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.ax = ax
            # self.ax.axis([-5,5,-5,5,0,5])

        self.ax.cla()
        # self.dynamic_model.state = self.euler2quad(self.state)
        frame = self.world_frame()
        self.ax.plot(frame[0, [0, 2]], frame[1, [0, 2]],
                     frame[2, [0, 2]], '-', c='cyan')[0]
        self.ax.plot(frame[0, [1, 3]], frame[1, [1, 3]],
                     frame[2, [1, 3]], '-', c='red')[0]
        self.ax.plot(frame[0, [4, 5]], frame[1, [4, 5]], frame[2, [
                     4, 5]], '-', c='blue', marker='o', markevery=2)[0]

        self.ax.plot([self.goal[0]], [self.goal[1]],
                     [self.goal[2]], '-o', c='red')

        # TODO: render obstacles

        self.ax.set_xlim([MIN_X, MAX_X])
        self.ax.set_ylim([MIN_Y, MAX_Y])
        self.ax.set_zlim([MIN_Z, MAX_Z])
        plt.pause(t)


def LQR_continuous_gain():
    A = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1/mass, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1/Ix, 0, 0],
        [0, 0, 1/Iy, 0],
        [0, 0, 0, 1/Iz],
    ])

    # Q = np.eye(12)*1.0
    Q = np.diag([1, 1.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    R = np.eye(4)*0.001

    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ S
    return K


def LQR_discrete_gain(dt):
    A = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    A = np.eye(12) + A*dt

    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1/mass, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1/Ix, 0, 0],
        [0, 0, 1/Iy, 0],
        [0, 0, 0, 1/Iz],
    ])

    B = B*dt

    # Q = np.eye(12)*1.0
    Q = np.diag([0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])*4
    R = np.diag([0.1,0.2,0.2,0.2])*0.001
    # Q = np.diag([1, 1.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # R = np.eye(4)*0.001

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


def LQR_control():
    env = QuadcopterEnv()
    env.reset()
    K = LQR_discrete_gain(env.dt)
    env.render()
    x_e, u_e = env.goal_, np.array([mass*g, 0, 0, 0])  # equalibrium
    while True:
        u = u_e - K @ (env.state_-x_e)
        obs, reward, done, info = env.step(env.normalize_u(u))
        env.render()
        if done: break


if __name__ == "__main__":
    LQR_control()
