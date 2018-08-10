#! /usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # goal-space=-.4to.4
    xs = np.array([4e3, 4e3, 5e3, 6e3])
    ys = np.array([4e-4, 2e-4, 2e-4, 2e-4])
    zs = np.array([512, 512, 512, 512])
    ax.scatter(xs, ys, zs, c='r')

    # goal-space=-.2to.2
    xs = np.array([3e3, 5e3, 1e4, 5e3, 5e3])
    ys = np.array([2e-4, 2e-4, 1e-4, 1e-4, 1e-4])
    zs = np.array([256, 256, 256, 256, 256])
    ax.scatter(xs, ys, zs, c='b')

    # goal-space=-.2to.2
    xs = np.array([3e3, 5e3, 1e4, 5e3, 5e3])
    ys = np.array([2e-4, 2e-4, 1e-4, 1e-4, 1e-4])
    zs = np.array([256, 256, 256, 256, 256])
    ax.scatter(xs, ys, zs, c='b')

    ax.set_xlabel('reward scale')
    ax.set_ylabel('learning rate')
    ax.set_zlabel('layer size')

    plt.save('best-params')
    plt.show()


if __name__ == '__main__':
    main()
