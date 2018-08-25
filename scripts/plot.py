#! /usr/bin/env python
import argparse
import csv
import itertools
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import numpy as np

from runs.database import DataBase
from runs.database import RunEntry
from runs.logger import Logger
from scripts.crawl_events import crawl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv-dir', nargs='*', type=PurePath)
    parser.add_argument('--components', default=None, nargs='+')
    args = parser.parse_args()

    fig = plt.figure()
    if len(args.components) == 1:
        projection = '2d'
    elif len(args.components) == 2:
        projection = '3d'
    else:
        raise RuntimeError('Must have 2 or 3 components')

    ax = fig.add_subplot(111, projection=projection)

    csv_files = args.csv_dir.glob('*.csv')
    colors = itertools.cycle('bgrcmkw')
    for path, color in zip(csv_files, colors):
        with path.open() as csv_file:
            table = csv.DictReader(csv_file)
            components = [[row[c] for row in table]
                          for c in args.components]
            ax.scatter(*components, c=color)

    ax.set_xlabel(args.components[0])
    ax.set_ylabel(args.components[1])
    try:
        ax.set_zlabel(args.components[2])
    except IndexError:
        pass

    plt.show()

    # # goal-space=-.4to.4
    # # rldl7: sort-runs .runs/tensorboard/multi-task-2d/goal_space=-.4to.4/ --smoothing=50
    # xs = np.array([3e3, 3e3, 6e3, 4e3, 5e3, 4e3])
    # ys = np.array([2e-4, 3e-4, 2e-4, 2e-4, 2e-4, 4e-4])
    # zs = np.array([512] * len(ys))
    # ax.scatter(xs, ys, zs, c='r')
    #
    # # goal-space=-.2to.2
    # # rldl7: sort-runs .runs/tensorboard/multi-task/geofence=.06/her-settings --smoothing=50
    # xs = np.array([5e3, 2e4, 1e4, 1e4, 1e4, 5e3, 1e4, 5e3, 5e3, 5e3, 5e3])
    # ys = np.array([2e-4, 5e-5, 5e-5, 5e-5, 1e-4, 2e-4, 1e-4, 5e-5, 1e-4, 1e-4, 1e-4])
    # zs = np.array([256] * len(ys))
    # ax.scatter(xs, ys, zs, c='b')
    #
    # # fixed-shift=-.2to.2
    # # rldl12: sort-runs .runs/tensorboard/shift/geofence=.06/bumpers --smoothing=50
    # xs = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 5e4])
    # ys = np.array([2e-4, 5e-5, 2e-4, 5e-5, 1e-4, 5e-5])
    # zs = np.array([256] * len(ys))
    # ax.scatter(xs, ys, zs, c='g', marker='x')
    #
    # # plt.save('best-params')
    # plt.show()


if __name__ == '__main__':
    main()
