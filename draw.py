#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@author: daishilong
@contact: daishilong1236@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from rigid import RectObs, RectRobot, CircleRobot

def plot_arrow(ax, x, y, yaw, length=1.5, width=0.3, color='k'): 
    arrow = ax.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            head_length=width, head_width=width, fc=color, ec=color)
    return [arrow]

def plot_robot(ax, rigidrobot, pose):
    rigidrobot.pose = pose[:3]
    outline = rigidrobot.outline()
    plot_outline, = plt.plot(outline[:,0], outline[:,1], c=rigidrobot.color)
    return [plot_outline]

def plot_rectobs(ax, obs):
    rect = patches.Rectangle(obs.rect[0], *(obs.rect[1]-obs.rect[0]), color=obs.color)
    return ax.add_patch(rect)

def plot_obs_list(ax, obs_list):
    collection = []
    for obs in obs_list:
        if isinstance(obs, RectObs):
            collection.append(plot_rectobs(ax, obs))
    return collection


def plot_problem_definition(ax, obs_list, rigidrobot, start, goal):
    """
    plot the obstacles, start and goal 

    Parameters
    ----------
    ax: 
        figure axis
    obs_list: array_like
        list of obstacles
    obs_size: float
        obstacle size
    start: array_like
        start state
    goal: array_like
        goal state
    
    Return
    ------
    collection: list
        a collection of matplotlib artists
    """
    collection = [] 
    ax_ob = plot_obs_list(ax, obs_list)
    start_mark = plot_arrow(ax, *start[:3], color='k')
    goal_mark = plot_arrow(ax, *goal[:3], color='b')
    collection += ax_ob + start_mark + goal_mark
    return collection

def draw_tree(robot_env, start, goal, tree, vertex=True, save_gif=True, fname='rrt_tree'):
    """Draw the tree built by the planner
    Args:
        robot_env (rl_planner.env.base_env.BaseEnv): the robot gym env
        start (numpy.ndarray)
        goal (numpy.ndarray)
        tree (list): list of nodes
        vertex (bool): nodes will be plotted if vertex is True 
        fname (str)
    """
    fig, ax = plt.subplots(figsize=(6,6))
    plt.axis([-22,22,-22,22])
    plt.xticks([])
    plt.yticks([])
    collection_list = [] # each entry is a collection
    tmp = plot_problem_definition(ax, robot_env.obs_list, robot_env.rigid_robot, start, goal)
    collection_list.append(tmp)
    for node in tree:
        if node.parent:
            tmp = tmp.copy()
            path = np.array(node.path[:])
            ax_path, = plt.plot(path[:,0], path[:,1], "-g", linewidth=0.6)
            tmp.append(ax_path)
            if vertex:   
                ax_node, = plt.plot(node.state[0], node.state[1], 'x', c='black', markersize=1.0)
                tmp.append(ax_node)
            collection_list.append(tmp)
            # plt.pause(2)   
    plt.savefig(fname+'.png')
    # plt.show()
    gif = anim.ArtistAnimation(fig, collection_list, interval=50)
    if save_gif:
        gif.save(fname+'.gif', writer = anim.PillowWriter(fps=4))


def draw_path(robot_env, start, goal, path, fname='rrt_path'):
    """Draw the planned path.
    Args:
        robot_env (rl_planner.env.base_env.BaseEnv): the robot gym env
        start (numpy.ndarray)
        goal (numpy.ndarray)
        path (list): the planned path
        fname (str)
    """
    fig, ax = plt.subplots(figsize=(6,6))
    plt.axis([-22,22,-22,22])
    plt.xticks([])
    plt.yticks([])
    collection_list = [] # each entry is a collection
    tmp = plot_problem_definition(ax, robot_env.obs_list, robot_env.rigid_robot, start, goal)
    array_path = np.array([state[:2] for state in path])
    plt.plot(array_path[:,0], array_path[:,1], c='k', linewidth=1.0)
    collection_list.append(tmp)

    for state in path:
        tmp_ = tmp.copy()
        robot_marker = plot_robot(ax, robot_env.rigid_robot, state[:3])
        tmp_ += robot_marker
        collection_list.append(tmp_)
    gif = anim.ArtistAnimation(fig, collection_list, interval=200)
    gif.save(fname+'.gif', writer = anim.PillowWriter(fps=5))



