
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import itertools


class param:
    # Vehicle config
    wheelbase = 2.33/2  # wheel base: front to rear axle [m]
    wheeldist = 1.85/2  # wheel dist: left to right wheel [m]
    v_w = 2.33/2  # vehicle width [m]
    r_b = 0.80/2  # rear to back [m]
    r_f = 3.15/2  # rear to front [m]
    t_r = 0.40/2  # tire radius [m]
    t_w = 0.30/2  # tire width [m]

    max_v = 2.0  # [m/s]
    min_v = -1.0  # [m/s]
    max_steer_angle = np.deg2rad(40)  # [rad]

    x_min = -20.0  # m
    x_max = 20.0  # m
    y_min = -20.0  # m
    y_max = 20.0  # m


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


def draw_car(ax, x, y, yaw, steer, color='black'):
    wheelbase = param.wheelbase  # wheel base: front to rear axle [m]
    wheeldist = param.wheeldist  # wheel dist: left to right wheel [m]
    v_w = param.v_w  # vehicle width [m]
    r_b = param.r_b  # rear to back [m]
    r_f = param.r_f  # rear to front [m]
    t_r = param.t_r  # tire radius [m]
    t_w = param.t_w  # tire width [m]
    car = np.array([[-r_b, -r_b, r_f, r_f, -r_b],
                    [v_w / 2, -v_w / 2, -v_w / 2, v_w / 2, v_w / 2]])

    wheel = np.array([[-t_r, -t_r, t_r, t_r, -t_r],
                      [t_w / 2, -t_w / 2, -t_w / 2, t_w / 2, t_w / 2]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])

    Rot2 = np.array([[np.cos(steer), np.sin(steer)],
                     [-np.sin(steer), np.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[wheelbase], [-wheeldist / 2]])
    flWheel += np.array([[wheelbase], [wheeldist / 2]])
    rrWheel[1, :] -= wheeldist / 2
    rlWheel[1, :] += wheeldist / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    ax.plot(car[0, :], car[1, :], color)
    ax.plot(frWheel[0, :], frWheel[1, :], color)
    ax.plot(rrWheel[0, :], rrWheel[1, :], color)
    ax.plot(flWheel[0, :], flWheel[1, :], color)
    ax.plot(rlWheel[0, :], rlWheel[1, :], color)
    Arrow(ax, x, y, yaw, 0.8 * wheelbase, color)


class car_points:
    def __init__(self):
        self.res = 0.1
        x_min_max = np.append(
            np.arange(param.r_b, param.r_f, self.res), param.r_f)
        y_min_max = np.append(
            np.arange(-param.v_w/2, param.v_w/2, self.res), param.v_w/2)
        self.points = np.array(list(itertools.product(x_min_max, y_min_max)))

    def get_points_world_frame(self, x, y, yaw):
        """Get points in world frame
        """
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        wPoints = self.points @ rot.T + np.array(x, y)
        return wPoints

def wrap_angle(angle):
    norm_angle = angle % (2 * np.pi)
    if norm_angle > np.pi:
        norm_angle -= 2 * np.pi
    return norm_angle

class Car(object):
    def __init__(self, **kwargs):
        self.car_points = car_points()
        self.obs_list = []

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
        state_dot[0] = input_u[0] * np.cos(state[2])
        state_dot[1] = input_u[0] * np.sin(state[2])
        state_dot[2] = input_u[0] / param.wheelbase * np.tan(input_u[1])
        return state_dot

    def propagate(self, start, control, duration):
        control = control.copy()
        control[0] = np.clip(control[0], param.min_v, param.max_v)
        control[1] = np.clip(control[1], -param.max_steer_angle, param.max_steer_angle)

        state = integrate.odeint(self.state_dot, start, [
                                 0, duration], args=(control,))[1]
        state[2] = wrap_angle(state[2])
        state[0] = np.clip(state[0], param.x_min, param.x_max)
        state[1] = np.clip(state[1], param.y_min, param.y_max)
        return state

    def valid_state(self, state):
        for obs in self.obs_list:
            if np.any(obs.points_in_obstacle(self.car_points.get_points_world_frame(*state))):
                return False
        return True

