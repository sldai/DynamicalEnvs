from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import itertools


class Obstacle(ABC):
    def __init__(self, color='black', **kwargs):
        super().__init__(**kwargs)
        self.color = color

    @abstractmethod
    def points_in_obstacle(self, points):
        """Check collision for a batch of points    
        Args:
            points (2d array): [[x1, y1], [x2, y2], ...]
        Raises:
            NotImplementedError: [description]
        Returns:
            1d array: [1, 1, 0, ...], 1 is collision, 0 is not
        """
        raise NotImplementedError()

    @abstractmethod
    def draw(self, ax):
        """Draw the obstacle shape in the axis of matplotlib
        Args:
            ax (pyplot ax): [description]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()


class RectangleObs(Obstacle):
    """AABB obstacle
    Args:
        Obstacle (base class): obstacles with color
    """

    def __init__(self, rect, **kwargs):
        """
        Parameters
        rect: [[left, bot], [right, top]]
        """
        super().__init__(**kwargs)
        assert np.any(rect[1] > rect[0]), 'Not a normal rectangle'
        self.rect = np.array(rect, dtype=np.float32)

    def points_in_obstacle(self, points):
        """
        Checking collision for a batch of points
        """
        c1 = np.logical_and(
            points[:, 0] >= self.rect[0, 0], points[:, 0] <= self.rect[1, 0])
        c2 = np.logical_and(
            points[:, 1] >= self.rect[0, 1], points[:, 1] <= self.rect[1, 1])
        return np.logical_and(c1, c2)

    def draw(self, ax):
        rect = patches.Rectangle(self.rect[0, :], *(self.rect[1, :] - self.rect[0, :]), color=self.color)
        return ax.add_patch(rect)


class CircleObs(Obstacle):
    """Round obstacles
    Args:
        Obstacle (base class): obstacles with color
    """

    def __init__(self, pos: np.ndarray, radius: float, **kwargs):
        """init
        Args:
            pos (np.ndarray): center position 
            radius (float): radius
        """
        super().__init__(**kwargs)
        self.pos = np.array(pos)
        self.radius = radius

    def points_in_obstacle(self, points):
        return np.linalg.norm(points - self.pos, axis=1) <= self.radius

    def draw(self, ax):
        circle = patches.Circle(self.pos, self.radius, color=self.color)
        return ax.add_patch(circle)


class SquareObs(Obstacle):
    """Square obstacle
    Args:
        Obstacle (base class ): obstacles with color
    """

    def __init__(self, pos: np.ndarray, size: float, **kwargs):
        """init
        Args:
            pos (np.ndarray): position of center
            size (float): size = width = length
        """
        super().__init__(**kwargs)
        self.pos = np.array(pos, dtype=np.float32)
        self.size = size

    def points_in_obstacle(self, points):
        c = np.abs(points - self.pos) <= self.size / 2
        return np.logical_and(c[:, 0], c[:, 1])

    def draw(self, ax):
        square = patches.Rectangle(self.pos - self.size / 2, self.size, self.size, color=self.color)
        return ax.add_patch(square)


class Environment(object):
    def __init__(self, **kwargs):
        self.boundary = np.array([
            [-5, 5],  # x range
            [-5, 5],  # y range
        ], dtype=np.float32)
        rect = RectangleObs(np.array([[-2, -1], [2, 1]]))

        circle = CircleObs((2, 1), 1, color='green')

        square = SquareObs((-2, -1), 1, color='cyan')
        self.obstacles = [rect, circle, square]
        
    def is_valid_points(self, points):
        """Check whether points are valid, i.e. not in collision and in the boundary
        Args:
            points (2d array):
        """
        valid = np.ones((len(points),), dtype=bool)

        # check boundary
        b_valid = np.logical_and(self.boundary[0, 0] <= points[:, 0],
                                 points[:, 0] <= self.boundary[0, 1])
        b_valid = np.logical_and(b_valid,
                                 self.boundary[1, 0] <= points[:, 1])
        b_valid = np.logical_and(b_valid,
                                 points[:, 1] <= self.boundary[1, 1])

        valid = np.logical_and(valid, b_valid)

        # colision 
        for ind, obs in enumerate(self.obstacles):
            valid = np.logical_and(valid, np.logical_not(obs.points_in_obstacle(points)))
        return valid

    def draw(self, ax):
        shapes = [obs.draw(ax) for obs in self.obstacles]
        return shapes

def get_obstacle_points(env):
    points = np.array(list(itertools.product(np.linspace(*env.boundary[0,:], num=100), np.linspace(*env.boundary[1,:], num =100))))
    valid = env.is_valid_points(points)

    non_free_points = points[np.logical_not(valid)]
    free_points = points[valid]
    return non_free_points