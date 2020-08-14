from abc import ABC, abstractmethod
import gym
import numpy as np

class BaseEnv(ABC, gym.Env):
    """The base environment speficies four major component for kinodynamic planning.
    state_bounds: boundaries in the state space
    cbounds: boundaries in the control space
    motion: propagate the dynamics 
    valid_state_check: collision checking
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.state_bounds = None
        self.cbounds = None
        

    @abstractmethod
    def motion(self, x, u, dt):
        """propagate the dynamic: x' = f(x,u)  
        """
        pass

    @abstractmethod
    def valid_state_check(self, state):
        """Valid state checking, if the state bounds are violated is firstly checked, 
        then do collision checking
        """
        return np.all(state-self.state_bounds[:,0]>=0) \
            and np.all(state-self.state_bounds[:,1]<=0)

    @abstractmethod
    def distance(self, state, goal):
        """Return the distance defined according to the environment
        """
        pass

    @abstractmethod
    def reach(self, state, goal):
        """
        Return True if the state reach the goal region
        """
        pass

    @abstractmethod
    def _obs(self): 
        """
        Return observations
        """
        pass
    
    @property
    def state_(self):
        return self.state.copy()

    @state_.setter
    def state_(self, v):
        self.state = v.copy()
    
    @property
    def goal_(self):
        return self.goal.copy()
        
    @goal_.setter
    def goal_(self, v):
        self.goal = v.copy()

