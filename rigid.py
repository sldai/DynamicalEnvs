
import numpy as np
from abc import ABC, abstractmethod
import os

def point_closest_point_on_line_segment(point, segment):
    """Calculates the point on the line segment that is closest
    to the specified point.
    This is similar to point_closest_point_on_line, except this
    is against the line segment of finite length. Whereas point_closest_point_on_line
    checks against a line of infinite length.
    :param numpy.array point: The point to check with.
    :param numpy.array line_segment: The finite line segment to check against.
    :rtype: numpy.array
    :return: The closest point on the line segment to the point.
    """
    # check if the line has any length
    rl = segment[1] - segment[0]
    squared_length = np.linalg.norm(rl)**2
    if squared_length == 0.0:
        return segment[0]

    rp = point - segment[0]
    # |a||b|cos(theta)/|b||b|
    dot = np.dot(rp, rl) / squared_length

    if dot < 0.0:
        return segment[0]
    elif dot > 1.0:
        return segment[1]

    # within segment
    # perform the same calculation as closest_point_on_line
    return segment[0] + (rl * dot)


def ray_intersect_aabb(ray, aabb):
    """Calculates the intersection point of a ray and an AABB using stab method
    :param numpy.array ray1: The ray to check.
    :param numpy.array aabb: The Axis-Aligned Bounding Box to check against.
    :rtype: numpy.array
    :return: Returns a vector if an intersection occurs.
        Returns None if no intersection occurs.
    """
    # this is basically "numpy.divide( 1.0, ray[ 1 ] )"
    # except we're trying to avoid a divide by zero warning
    # so where the ray direction value is 0.0, just use infinity
    # which is what we want anyway
    direction = ray[1]
    dir_fraction = np.empty(2, dtype=ray.dtype)
    dir_fraction[direction == 0.0] = 1e300
    dir_fraction[direction != 0.0] = np.divide(
        1.0, direction[direction != 0.0])

    t1 = (aabb[0, 0] - ray[0, 0]) * dir_fraction[0]
    t2 = (aabb[1, 0] - ray[0, 0]) * dir_fraction[0]
    t3 = (aabb[0, 1] - ray[0, 1]) * dir_fraction[1]
    t4 = (aabb[1, 1] - ray[0, 1]) * dir_fraction[1]

    tmin = max(min(t1, t2), min(t3, t4))
    tmax = min(max(t1, t2), max(t3, t4))

    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    if tmax < 0:
        return None

    # if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax:
        return None

    # t is the distance from the ray point
    # to intersection

    t = min(x for x in [tmin, tmax] if x >= 0)
    point = ray[0] + (ray[1] * t)
    return point


def line_segment_intersect_aabb(line_seg, aabb):
    """
    Calculates the intersection point of a line segment and an AABB
    :param line_seg: [n1, n2]
    :param aabb: [left_bot, right_top]
    """

    # check endpoint in rectangle
    s = line_seg[0]
    # if (aabb[0,0]<=s[0]<=aabb[1,0]) and (aabb[0,1]<=s[1]<=aabb[1,1]) \
    #     and (aabb[0,2]<=s[2]<=aabb[1,2]):
    #     return True

    # s is outside, check intersection
    ray = line_seg.copy()
    ray[1] = (line_seg[1] - line_seg[0]) / \
        np.linalg.norm((line_seg[1] - line_seg[0]))
    point = ray_intersect_aabb(ray, aabb)

    if point is None:  # not intersect
        return None
    else:
        # intersect, but need to check whether intersection lays on the line segment
        if np.linalg.norm(point-line_seg[0]) <= np.linalg.norm(line_seg[1]-line_seg[0]):
            # point is on the line segment
            return point
        else:
            # point is outside of the line segment
            return None

def aabb_intersect_aabb(aabb1, aabb2):
    for i in range(2):
        if np.all(aabb1[:,i] - aabb2[0,i]<0) or np.all(aabb1[:,i] - aabb2[1,i]>0):
            return False
    return True


class Obstacle(ABC):
    def __init__(self, color='black'):
        self.color = color

    @abstractmethod
    def point_in_obstacle(self, point):
        raise NotImplementedError()

    @abstractmethod
    def dis2point(self, point):
        raise NotImplementedError()


class RectObs(Obstacle):
    def __init__(self, rect, color='black'):
        """
        Parameters
        rect: [[left, bot], [right, top]]
        """
        super().__init__(color=color)
        assert np.any(rect[1] > rect[0]), 'Not a normal rectangle'
        self.rect = np.array(rect, dtype=np.float32)

    def point_in_obstacle(self, point):
        """
        Parameters:
        point: 2d array
        """
        x, y = point
        if self.rect[0, 0] <= x <= self.rect[1, 0] and self.rect[0, 1] <= y <= self.rect[1, 1]:
            return True
        return False

    def points_in_obstacle(self, points):
        """
        Checking collision for a batch of points
        """
        c1 = np.logical_and(
            points[:, 0] >= self.rect[0, 0], points[:, 0] <= self.rect[1, 0])
        c2 = np.logical_and(
            points[:, 1] >= self.rect[0, 1], points[:, 1] <= self.rect[1, 1])
        return np.logical_and(c1, c2)

    def dis2point(self, point):
        left, bottom, right, top = self.rect.flatten()
        # get the four edges of the rectangle
        edge1 = np.array([[left, bottom], [left, top]])
        edge2 = np.array([[left, top], [right, top]])
        edge3 = np.array([[right, top], [right, bottom]])
        edge4 = np.array([[right, bottom], [left, bottom]])

        dis = np.inf
        for edge in [edge1, edge2, edge3, edge4]:
            point_ = point_closest_point_on_line_segment(point, edge)
            dis = min(dis, np.linalg.norm(point-point_))
        return dis
    

class CubeObs(Obstacle):
    def __init__(self, pos, size, color='black'):
        """Cube obstacle for 3D environment

        Arguments:
            Obstacle  -- base class
            pos {array} -- 3D position
            size {float} -- cube size

        Keyword Arguments:
            color {str} -- color (default: {'black'})
        """
        self.color = color
        self.pos = pos.copy()
        self.size = size

    def point_in_obstacle(self, point):
        return self.points_in_obstacle(point.reshape((1,-1)))[0]
    
    def points_in_obstacle(self, points):
        """Collision check for a batch of points

        Arguments:
            points {2d array} -- [[x1,y1,z1], [x2,y2,z2], ...]
        
        Returns:
            [1d array] -- True is collision       
        """
        c = np.abs(points - self.pos) <= self.size/2
        
        return np.logical_and(np.logical_and(c[:,0], c[:,1]), c[:,2])

    def dis2point(self, point):
        raise NotImplementedError()

    def draw(self, ax):
        x = [
            self.pos[0] - self.size/2,
            self.pos[0] + self.size/2,
            self.pos[0] + self.size/2,
            self.pos[0] - self.size/2,
            self.pos[0] - self.size/2,
            self.pos[0] + self.size/2,
            self.pos[0] + self.size/2,
            self.pos[0] - self.size/2
            ]
        y = [
            self.pos[1] - self.size/2,
            self.pos[1] - self.size/2,
            self.pos[1] + self.size/2,
            self.pos[1] + self.size/2,
            self.pos[1] - self.size/2,
            self.pos[1] - self.size/2,
            self.pos[1] + self.size/2,
            self.pos[1] + self.size/2
        ] 
        z = [
            self.pos[2] - self.size/2,
            self.pos[2] - self.size/2,
            self.pos[2] - self.size/2,
            self.pos[2] - self.size/2,
            self.pos[2] + self.size/2,
            self.pos[2] + self.size/2,
            self.pos[2] + self.size/2,
            self.pos[2] + self.size/2
        ]
        points = np.array(list(zip(x, y, z)))
        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
        [points[4],points[5],points[6],points[7]], 
        [points[0],points[1],points[5],points[4]], 
        [points[2],points[3],points[7],points[6]], 
        [points[1],points[2],points[6],points[5]],
        [points[4],points[7],points[3],points[0]]]
        return ax.add_collection3d(Poly3DCollection(verts, 
                            facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    



def T_transform2d(aTb, bP):
    """
    aTb: transform with rotation, translation
    bP: non-holonomic coordinates in b
    return: non-holonomic coordinates in a
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


######################## robot class ####################


class Rigid(ABC):
    """
    The rigid robot base class
    """

    def __init__(self, pose=np.zeros(3), color='black'):
        self.color = color
        self.pose = np.array(pose, dtype=float)

    @staticmethod
    def transformation2world(x, pose):
        """
        x: [[x1,y1],
            [x2,y2],
            ...]
            stacked coordinates 
        """
        # from body to world
        wRb = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                        [np.sin(pose[2]), np.cos(pose[2])]])
        wTb = np.block([[wRb, pose[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        return T_transform2d(wTb, x)


    @staticmethod
    def transformation_from_world(x, pose):
        # from world to body
        wRb = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                        [np.sin(pose[2]), np.cos(pose[2])]])
        wTb = np.block([[wRb, pose[:2].reshape((-1, 1))],
                        [np.zeros((1, 2)), 1]])
        return T_transform2d(np.linalg.inv(wTb), x)
        


    @abstractmethod
    def dis2obs(self, obstacle):
        """
        Return -1 if collsion, else the distance to the obstacle (may be approximate)
        """
        raise NotImplementedError()

    @abstractmethod
    def collision_obs(self, obstacle):
        """Return True if collision
        """




class RectRobot(Rigid):
    def __init__(self, rect, pose=np.zeros(3), color='black'):
        """
        Parameters
        ----------
        rect: [[left, bot], [right, top]]
        pose: [x,y,theta]
        """
        super().__init__(pose, color)
        self.rect = np.array(rect, dtype=np.float32)

    def dis2obs(self, obstacle):
        dis = np.inf
        if isinstance(obstacle, RectObs):
            left, bottom, right, top = self.rect.flatten()
            vertices = np.array([[left, bottom],
                                 [left, top],
                                 [right, top],
                                 [right, bottom],
                                 ])
            vertices = self.transformation2world(vertices, self.pose)

            # collision when any vertex is in the obstacle
            for x in vertices:
                if obstacle.point_in_obstacle(x):
                    return -1

            # check intersection
            for ind in range(len(vertices)):
                s = vertices[ind]
                g = vertices[(ind+1) % len(vertices)]
                line_seg = np.array([s, g])
                p = line_segment_intersect_aabb(line_seg, obstacle.rect)
                if p is not None:  # intersection
                    return -1

            # no collision, calculate distance
            dis = np.inf
            # distance from vertices of the robot to the obstacle
            for x in vertices:
                dis = min(dis, obstacle.dis2point(x))

            # transform things from world frame to robot body frame
            # distance from vertices of the obstacle to the robot
            robot_obs = RectObs(self.rect)

            left, bottom, right, top = obstacle.rect.flatten()
            vertices = np.array([[left, bottom],
                                 [left, top],
                                 [right, top],
                                 [right, bottom],
                                 ])
            vertices = self.transformation_from_world(vertices, self.pose)
            for x in vertices:
                dis = min(dis, robot_obs.dis2point(x))
            
            # the extra term accounts the noise
            return dis if dis > 0.05 else -1
        else:
            raise NotImplementedError()

    def collision_obs(self, obstacle):
        if isinstance(obstacle, RectObs):
            left, bottom, right, top = self.rect.flatten()
            vertices = np.array([[left, bottom],
                                 [left, top],
                                 [right, top],
                                 [right, bottom],
                                 ])
            vertices = self.transformation2world(vertices, self.pose)

            # collision when any vertex is in the obstacle
            for x in vertices:
                if obstacle.point_in_obstacle(x):
                    return True

            # check intersection
            for ind in range(len(vertices)):
                s = vertices[ind]
                g = vertices[(ind+1) % len(vertices)]
                line_seg = np.array([s, g])
                p = line_segment_intersect_aabb(line_seg, obstacle.rect)
                if p is not None:  # intersection
                    return True
            return False

    def outline(self):
        """
        Return the robot outline
        """
        left, bottom, right, top = self.rect.flatten()
        vertices = np.array([[left, bottom],
                             [left, top],
                             [right, top],
                             [right, bottom],
                             [left, bottom]
                             ])
        vertices = self.transformation2world(vertices, self.pose)

        return vertices


class CircleRobot(Rigid):
    def __init__(self, radius, pose=np.zeros(3), color='black'):
        super().__init__(pose, color)
        self.radius = radius

    def dis2obs(self, obstacle):
        if isinstance(obstacle, RectObs):
            center = self.transformation2world(np.zeros((1, 2)), self.pose)[0]

            if obstacle.point_in_obstacle(center):
                return -1
            dis = obstacle.dis2point(center)-self.radius
            
            # the extra term accounts the noise
            return dis if dis> 0.05 else -1
        else:
            raise NotImplementedError()
        
    def collision_obs(self, obstacle):
        return self.dis2obs(obstacle)==-1

    def outline(self):
        num = 20
        thetas = np.linspace(-np.pi, np.pi, num=num)
        circle = self.pose[:2].reshape(
            1, -1) + self.radius*np.array([np.cos(thetas), np.sin(thetas)]).T
        return circle


class Vehicle(RectRobot):
    def __init__(self, d, rect, pose=np.zeros(3), color='black'):
        super().__init__(rect, pose, color)
        self.d = d

class CartPole(object):
    def __init__(self, rect, l, x, theta, cart_color='blue', pole_color='red'):
        super().__init__()
        self.cart = RectRobot(rect, pose=np.array([x,0.0,0.0]), color=cart_color)
        self.pole_color = pole_color
        self.l = l
        self.theta = theta

    def set_config(self, x, theta):
        self.cart.pose[0] = x
        self.theta = theta

    def get_pole(self):
        pole = np.zeros([2,2])
        pole[0] = self.cart.pose[:2]
        pole[1,0] = pole[0,0] + self.l*np.cos(self.theta-np.pi/2)
        pole[1,1] = pole[0,1] + self.l*np.sin(self.theta-np.pi/2)
        return pole

    def collision_obs(self, obstacle):
        if isinstance(obstacle, RectObs):
            pole = self.get_pole()
            collision = self.cart.collision_obs(obstacle) \
                        or line_segment_intersect_aabb(pole, obstacle.rect) is not None
            return collision

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
MIN_X = -1
MAX_X = 1
MIN_Y = -1
MAX_Y = 1
MIN_Z = 0
MAX_Z = 2
import pickle
if __name__ == "__main__":

    size = 0.4
    for i in range(1):
        obs = []
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(7):

            pos = np.random.uniform(low=np.array([-1, -1, 0.0])+size/2, high=np.array([1, 1, 2.0])-size/2)
            cube = CubeObs(pos=pos,size=size)
            obs.append(cube)
            cube.draw(ax)
        ax.set_xlim([MIN_X, MAX_X])
        ax.set_ylim([MIN_Y, MAX_Y])
        ax.set_zlim([MIN_Z, MAX_Z])
        obs_list = [{'pos': o.pos, 'size': o.size, 'color': o.color} for o in obs]
        pickle.dump(obs_list, open('quadcopter_obstacles.pkl','wb'))
        plt.savefig('quadcopter_obstacles.png')
    
    plt.show()


    # draw
