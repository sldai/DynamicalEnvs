import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/..')
from system.quadrotor import *
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

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

class QuadrotorEnv(gym):
    def __init__(**kwargs):
        super().__init__(**kwargs)
        self.system = Quadrotor()
        self.state_space = np.array(
            []
        )

        self.control_space = np.array([

        ])

        self.obser
    
    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

