import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/..')
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import itertools


class param:
    # Vehicle config
    v_r = 0.6/2  # vehicle radius [m]
    wheelbase = 0.4/2  # wheel base: center to rear axle [m]
    wheeldist = 0.8/2  # wheel dist: left to right wheel [m]
    t_r = 0.40/2/2  # tire radius [m]
    t_w = 0.30/2/2  # tire width [m]

    max_v = 0.5  # [m/s]
    min_v = -0.2  # [m/s]
    max_w = np.pi  # [rad/s]
    max_acc_v = 1.0  # [m/s^2]
    max_acc_w = np.pi  # [rad/s^2]

    x_min = -20.0  # [m]
    x_max = 20.0  # [m]
    y_min = -20.0  # [m]
    y_max = 20.0  # [m]

    dt = 0.2  # duration of one control step [s]
    integration_dt = 2e-2  # for integration [s]

    goal_radius = 0.5  # goal region [m]

class Arrow:
    def __init__(self, ax, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.3 * L
        w = 2
        PI = np.pi
        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + PI - angle
        theta_hat_R = theta + PI + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        ax.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_L],
                [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_R],
                [y_hat_start, y_hat_end_R], color=c, linewidth=w)


def draw_base(ax, x, y, yaw, color='black'):
    angles = np.linspace(-np.pi, np.pi)
    base = np.zeros((len(angles), 2))
    base[:, 0] = param.v_r * np.cos(angles)
    base[:, 1] = param.v_r * np.sin(angles)

    wheel = np.array([[-param.t_r, -param.t_r, param.t_r, param.t_r, -param.t_r],
                      [param.t_w / 2, -param.t_w / 2, -param.t_w / 2, param.t_w / 2, param.t_w / 2]])

    lWheel = wheel.copy()
    rWheel = wheel.copy()

    Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])

    rWheel[1, :] -= param.wheeldist / 2
    lWheel[1, :] += param.wheeldist / 2

    rWheel = np.dot(Rot1, rWheel)
    lWheel = np.dot(Rot1, lWheel)
    base = base @ Rot1.T

    rWheel += np.array([[x], [y]])
    lWheel += np.array([[x], [y]])
    base += np.array([x, y])

    ax.plot(base[:, 0], base[:, 1], color)
    ax.plot(lWheel[0, :], lWheel[1, :], color)
    ax.plot(rWheel[0, :], rWheel[1, :], color)
    Arrow(ax, x-0.5 * np.cos(yaw) * param.wheelbase, y-0.5 * np.sin(yaw)
          * param.wheelbase, yaw, 1.3 * param.wheelbase, color)


class base_points:
    def __init__(self, res=0.1, radius=param.v_r):
        self.res = res
        angles = np.linspace(-np.pi, np.pi, 50)
        base = np.zeros((len(angles), 2))
        base[:, 0] = radius * np.cos(angles)
        base[:, 1] = radius * np.sin(angles)
        self.points = base

    def get_points_world_frame(self, x, y, yaw):
        """Get points in world frame
        """
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        wPoints = self.points @ rot.T + np.array([x, y])
        return wPoints


def wrap_angle(angle):
    norm_angle = angle % (2 * np.pi)
    if norm_angle > np.pi:
        norm_angle -= 2 * np.pi
    return norm_angle

class DifferentialDrive(object):
    def __init__(self, **kwargs):
        self.base_points = base_points()
        self.obs_list = []
    
    def set_obs_list(self, obs_list):
        self.obs_list = obs_list
        
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
        state_dot[2] = state[4]
        state_dot[3] = input_u[0]
        state_dot[4] = input_u[1]
        return state_dot

    def propagate(self, start, control, duration):
        state = start.copy()
        control = control.copy()
        control[0] = np.clip(control[0], -param.max_acc_v, param.max_acc_v)
        control[1] = np.clip(
            control[1], -param.max_acc_w, param.max_acc_w)
        step_num = int(duration/param.integration_dt)
        
        for t in range(step_num):
            state += self.state_dot(state, 0, control) * param.integration_dt 
            state[2] = wrap_angle(state[2])
            state[0] = np.clip(state[0], param.x_min, param.x_max)
            state[1] = np.clip(state[1], param.y_min, param.y_max)
            state[3] = np.clip(state[3], param.min_v, param.max_v)
            state[4] = np.clip(state[4], -param.max_w, param.max_w)
        return state

    def valid_state(self, state):
        for obs in self.obs_list:
            if np.any(obs.points_in_obstacle(self.base_points.get_points_world_frame(*state[:3]))):
                return False
        return True

