#! /usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # goal-space=-.4to.4
    # rldl7: sort-runs .runs/tensorboard/multi-task-2d/goal_space=-.4to.4/ --smoothing=50
    xs = np.array([3e3, 3e3, 6e3, 4e3, 5e3, 4e3])
    ys = np.array([2e-4, 3e-4, 2e-4, 2e-4, 2e-4, 4e-4])
    zs = np.array([512] * len(ys))
    ax.scatter(xs, ys, zs, c='r')

    # goal-space=-.2to.2
    # rldl7: sort-runs .runs/tensorboard/multi-task/geofence=.06/her-settings --smoothing=50
    xs = np.array([5e3, 2e4, 1e4, 1e4, 1e4, 5e3, 1e4, 5e3, 5e3, 5e3, 5e3])
    ys = np.array([2e-4, 5e-5, 5e-5, 5e-5, 1e-4, 2e-4, 1e-4, 5e-5, 1e-4, 1e-4, 1e-4])
    zs = np.array([256] * len(ys))
    ax.scatter(xs, ys, zs, c='b')

    # fixed-shift=-.2to.2
    # rldl12: sort-runs .runs/tensorboard/shift/geofence=.06/bumpers --smoothing=50
    xs = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 5e4])
    ys = np.array([2e-4, 5e-5, 2e-4, 5e-5, 1e-4, 5e-5])
    zs = np.array([256] * len(ys))
    ax.scatter(xs, ys, zs, c='g', marker='x')

    ax.set_xlabel('reward scale')
    ax.set_ylabel('learning rate')
    ax.set_zlabel('layer size')

    # plt.save('best-params')
    plt.show()


if __name__ == '__main__':
    main()
