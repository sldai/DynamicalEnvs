import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/..')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_line_3d(ax, p, p_index, color='b', alpha=1):
    for p_i in p_index:
        ax.plot3D(p[p_i, 0], p[p_i, 1], p[p_i, 2], c=color, alpha=alpha)
        
def centered_box_to_points_3d(center, size):
    half_size = [s/2 for s in size]
    direction, p = [1, -1], []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([center[di] + [x_d, y_d, z_d][di] * half_size[0] for di in range(3)])
    return p

def rot_frame_3d(state, frame_size=0.25):
    b, c, d, a = state[3:7]
    rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                        [2 * b * c - 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d + 2 * a * b],
                        [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
    quadrotor_frame = np.array([[frame_size, 0, 0],
                                 [0, frame_size, 0],
                                 [-frame_size, 0, 0],
                                 [0, -frame_size, 0]]).T
    quadrotor_frame = rot_mat @ quadrotor_frame + state[:3].reshape(-1, 1)
    return quadrotor_frame

def q_to_points_3d(state):
    quadrotor_frame = rot_frame_3d(state)   
    max_min, direction = [np.max(quadrotor_frame, axis=1), np.min(quadrotor_frame, axis=1)], [1, 0]
    p = []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([max_min[x_d][0], max_min[y_d][1], max_min[z_d][2]])
    return np.array(p)

def draw_box_3d(ax, p, color='b', alpha=1, surface_color='blue', linewidths=1, edgecolors='k'):
    index_lists = [[[0, 4], [4, 6], [6, 2], [2, 0], [0, 1], [1, 5], [5, 7], [7, 3], [3, 1], [1, 5]],
                  [[4, 5]],
                  [[6, 7]],
                  [[2, 3]]]
    for p_i in index_lists:
        draw_line_3d(ax, np.array(p), p_i, color=color, alpha=alpha)
    edges = [[p[e_i] for e_i in f_i] for f_i in [[0, 1, 5, 4],
                                                 [4, 5, 7, 6],
                                                 [6, 7, 3, 2],
                                                 [2, 0, 1, 3],
                                                 [2, 0, 4, 6],
                                                 [3, 1, 5, 7]]]
    faces = Poly3DCollection(edges, linewidths=linewidths, edgecolors=edgecolors)
    faces.set_facecolor(surface_color)
    faces.set_alpha(0.1)
    ax.add_collection3d(faces)

def draw_quadrotor(ax, state, color='orange'):
    """state format: [x, y, z, quat_x, quat_y, quat_z, w, ...]
    """
    draw_box_3d(ax, q_to_points_3d(state), alpha=0.3, surface_color=color, linewidths=0.)

class Quadrotor(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MIN_C1 = -15.
        self.MAX_C1 = -5.
        self.MIN_C = -1.
        self.MAX_C = 1.
        self.MIN_X = -5
        self.MAX_X = 5
        self.MIN_V = -1.
        self.MAX_V = 1.
        self.MIN_W = -1.
        self.MAX_W = 1.
        self.MASS_INV = 1.
        self.BETA = 1.
        self.EPS = 2.107342e-08
        self.obs_list = []
        self.radius = 0.25
        self.width = 1.0
        self.integration_step = 2e-2

    def enforce_bounds_quaternion(self, qstate):
        # enforce quaternion
        # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
        # [x, y, z, w]
        nrmSqr = qstate[0]*qstate[0] + qstate[1]*qstate[1] + qstate[2]*qstate[2] + qstate[3]*qstate[3]
        nrmsq = np.sqrt(nrmSqr) if (np.abs(nrmSqr - 1.0) > 1e-6) else 1.0
        error = np.abs(1.0 - nrmsq)
        if error < self.EPS:
            scale = 2.0 / (1.0 + nrmsq)
            qstate *= scale
        else:
            if nrmsq < 1e-6:
                qstate[:] = 0
                qstate[3] = 1
            else:
                scale = 1.0 / np.sqrt(nrmsq)
                qstate *= scale
        return qstate

    def enforce_bounds_quaternion_vec(self, pose):
        # enforce quaternion
        # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
        # [x, y, z, w]
        nrmsq = np.sum(pose ** 2, axis=1)
        ind = np.abs(1.0 - nrmsq) < self.EPS
        pose[ind, :]  *= 2.0 / (1.0 + np.expand_dims(nrmsq[ind], axis=1))
        ind = nrmsq < 1e-6
        pose[ind, 0:3] = 0
        pose[ind, 3] = 1
        pose *= 1.0 / (np.expand_dims(np.sqrt(nrmsq), axis=1) + self.EPS)
        return pose
    
    def _compute_derivatives(self, q, u):
        qdot = np.zeros(q.shape)
        qdot[0:3] = q[7:10]
        qomega = np.zeros(4) #[ x, y, z, w,]
        qomega[0:3] = 0.5 * q[10:13]
        qomega = self.enforce_bounds_quaternion(qomega)
        delta = q[3] * qomega[0] + q[4] * qomega[1] + q[5] * qomega[2]
        qdot[3:7] = qomega - delta * q[3:7]
        qdot[7] = self.MASS_INV * (-2 * u[0] * (q[6] * q[4] + q[3] * q[5]) - self.BETA * q[7])
        qdot[8] = self.MASS_INV * (-2 * u[0] * (q[4] * q[5] - q[6] * q[3]) - self.BETA * q[8])
        qdot[9] = self.MASS_INV * (-u[0] * (q[6] * q[6] - q[3] * q[3] - q[4] * q[4] + q[5] * q[5]) - self.BETA * q[9]) - 9.81
        qdot[10:13] = u[1:4]
        return qdot
    
    def _compute_derivatives_vec(self, q, u):
        qdot = np.zeros(q.shape)
        qdot[:, 0:3] = q[:, 7:10]
        qomega = np.zeros((q.shape[0], 4)) #[ x, y, z, w,]
        qomega[:, 0:3] = 0.5 * q[:, 10:13]
        qomega = self.enforce_bounds_quaternion_vec(qomega)
        delta = q[:, 3] * qomega[:, 0] + q[:, 4] * qomega[:, 1] + q[:, 5] * qomega[:, 2]
        qdot[:, 3:7] = qomega - np.expand_dims(delta, axis=1) * q[:, 3:7]
        qdot[:, 7] = self.MASS_INV * (-2 * u[:, 0] * (q[:, 6] * q[:, 4] + q[:, 3] * q[:, 5]) - self.BETA * q[:, 7])
        qdot[:, 8] = self.MASS_INV * (-2 * u[:, 0] * (q[:, 4] * q[:, 5] - q[:, 6] * q[:, 3]) - self.BETA * q[:, 8])
        qdot[:, 9] = self.MASS_INV * (-u[:, 0] * (q[:, 6] * q[:, 6] - q[:, 3] * q[:, 3] - q[:, 4] * q[:, 4] + q[:, 5] * q[:, 5]) - self.BETA * q[:, 9]) - 9.81
        qdot[:, 10:13] = u[:, 1:4]
        return qdot

    def propagate(self, start_state, control, duration):
        '''
        control (n_sample)
        t is (n_sample)
        # control in [NS, NC=4], t in [NS]
        '''
        steps = int(duration/self.integration_step)
        q = start_state.copy()
        control[0] = np.clip(control[0], self.MIN_C1, self.MAX_C1)
        control[1] = np.clip(control[1], self.MIN_C, self.MAX_C)
        control[2] = np.clip(control[2], self.MIN_C, self.MAX_C)
        control[3] = np.clip(control[3], self.MIN_C, self.MAX_C)
        q[3:7] = self.enforce_bounds_quaternion(q[3:7])
        for t in range(0, steps):
            q += self.integration_step * self._compute_derivatives(q, control)
            q[:3] = np.clip(q[:3], self.MIN_X, self.MAX_X)
            q[7:11] = np.clip(q[7:11], self.MIN_V, self.MAX_V)
            q[10:13] = np.clip(q[10:13], self.MIN_W, self.MAX_W)
            q[3:7] = self.enforce_bounds_quaternion(q[3:7])
        return q


    def propagate_vec(self, start_state, control, t, integration_step, direction=1):
        '''
        control (n_sample)
        t is (n_sample)
        # control in [NS, NC=4], t in [NS]
        '''
        q = start_state
        control[:, 0] = np.clip(control[:, 0], self.MIN_C1, self.MAX_C1)
        control[:, 1] = np.clip(control[:, 1], self.MIN_C, self.MAX_C)
        control[:, 2] = np.clip(control[:, 2], self.MIN_C, self.MAX_C)
        control[:, 3] = np.clip(control[:, 3], self.MIN_C, self.MAX_C)
        q[:, 3:7] = self.enforce_bounds_quaternion_vec(q[:,3:7])
        t_max = np.max(t)
        for t_curr in np.arange(0, t_max + integration_step, integration_step):
            q[ t >= t_curr, :] += direction * integration_step * self._compute_derivatives_vec(q[t >= t_curr, :], control[t >= t_curr, :])
            q[:, 7:11] = np.clip(q[:, 7:11], self.MIN_V, self.MAX_V)
            q[:, 10:13] = np.clip(q[:, 10:13], self.MIN_W, self.MAX_W)
            q[:, 3:7] = self.enforce_bounds_quaternion_vec(q[:, 3:7])
        return q

    def valid_state(self, state):
        for obs in self.obs_list:
            corners = centered_box_to_points_3d(center=obs, size=[self.width]*3)
            obs_min_max = [np.min(corners, axis=0), np.max(corners, axis=0)]
            quadrotor_frame = rot_frame_3d(state, self.radius)   
            quadrotor_min_max = [np.min(quadrotor_frame, axis=1), np.max(quadrotor_frame, axis=1)]
            if quadrotor_min_max[0][0] <= obs_min_max[1][0] and quadrotor_min_max[1][0] >= obs_min_max[0][0] and\
                quadrotor_min_max[0][1] <= obs_min_max[1][1] and quadrotor_min_max[1][1] >= obs_min_max[0][1] and\
                quadrotor_min_max[0][2] <= obs_min_max[1][2] and quadrotor_min_max[1][2] >= obs_min_max[0][2]:
                    return False
        return True

