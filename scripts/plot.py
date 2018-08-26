#! /usr/bin/env python
import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import PurePath, Path
from typing import List, Dict
import numpy as np

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sac.utils import softmax, parse_double


def parse_double(string):
    if string is None:
        return
    a, b = map(float, string.split('x'))
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=Path)
    parser.add_argument('--components', default=None, nargs='+')
    parser.add_argument('--delimiter', default='|')
    parser.add_argument('--plot-path', default=Path('/tmp', 'plot.png'))
    parser.add_argument('--weight-key')
    parser.add_argument('--figsize', type=parse_double)
    args = parser.parse_args()

    plot(keys=args.components,
         delimiter=args.delimiter,
         csv_dir=args.csv_dir,
         plot_path=args.plot_path,
         weight_key=args.weight_key,
         figsize=args.figsize)

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


def collect_components(csv_path: Path, keys: List[str], delimiter: str):
    with csv_path.open() as csv_file:
        table = csv.DictReader(csv_file, delimiter=delimiter)
        values = defaultdict(list)
        for row in table:
            for key in keys:
                values[key].append(float(row[key]))
        return values


def subplot(components: Dict[str, List[float]], keys: List[str],
            fig, R, C, P, color: str = 'c'):
    if len(keys) == 2:
        projection = '2d'
    elif len(keys) > 2:
        projection = '3d'
    else:
        raise RuntimeError('Must have 2 or 3 components')

    ax = fig.add_subplot(R, C, P, projection=projection)
    ax.scatter(*[components[k] for k in keys], c=color)
    ax.set_xlabel(keys[0])
    ax.set_ylabel(keys[1])
    try:
        ax.set_zlabel(keys[2])
    except IndexError:
        pass


def plot(csv_dir,
         delimiter,
         keys,
         plot_path,
         weight_key,
         figsize):
    # default keys
    if keys is None:
        if weight_key == 'reward':
            keys = ['learning_rate', 'reward_scale', 'grad_clip']
        else:
            keys = ['learning_rate', 'reward_scale', 'reward']

    csv_paths = list(csv_dir.glob('*.csv'))
    # colors = itertools.cycle('bgrcmkw')

    if weight_key:
        components = {k: [] for k in keys}
        keys.append(weight_key)

    fig = plt.figure(figsize=figsize)
    for i, csv_path in enumerate(csv_paths):
        components = collect_components(csv_path=csv_path,
                                        keys=keys,
                                        delimiter=delimiter)
        if weight_key:
            weights = softmax(np.array(components[weight_key]))
            for k, v in components.items():
                components[k].append(np.dot(weights, np.array(v)))
        else:
            subplot(components=components, keys=keys, fig=fig,
                    R=len(csv_paths), C=1, P=i + 1)

    if weight_key:
        subplot(components=components, keys=keys, fig=fig,
                R=1, C=1, P=1)
    plt.savefig(plot_path)


if __name__ == '__main__':
    main()
