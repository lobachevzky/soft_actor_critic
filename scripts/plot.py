#! /usr/bin/env python
import argparse
import csv
import itertools
from pathlib import PurePath, Path

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=Path)
    parser.add_argument('--components', default=None, nargs='+')
    parser.add_argument('--delimiter', default='|')
    parser.add_argument('--plot-path', default=Path('/tmp', 'plot.png'))
    args = parser.parse_args()

    plot(component_names=args.components,
         delimiter=args.delimiter,
         csv_dir=args.csv_dir,
         plot_path=args.plot_path)

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


def plot(csv_dir,
         delimiter,
         component_names,
         plot_path):
    if component_names is None:
        component_names = ['learning_rate', 'reward_scale', 'reward']
    if len(component_names) == 2:
        projection = '2d'
    elif len(component_names) > 2:
        projection = '3d'
    else:
        raise RuntimeError('Must have 2 or 3 components')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    csv_paths = csv_dir.glob('*.csv')
    colors = itertools.cycle('bgrcmkw')
    for csv_path, color in zip(csv_paths, colors):
        with csv_path.open() as csv_file:
            table = csv.DictReader(csv_file, delimiter=delimiter)
            components = zip(*[[float(row[c]) for c in component_names]
                               for row in table])
            ax.scatter(*components, c=color)
    ax.set_xlabel(component_names[0])
    ax.set_ylabel(component_names[1])
    try:
        ax.set_zlabel(component_names[2])
    except IndexError:
        pass
    plt.savefig(plot_path)


if __name__ == '__main__':
    main()
