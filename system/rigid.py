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
