import sys 
import os 
sys.path.append('')
from system.quadrotor import Quadrotor
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
import gym

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

