#! /usr/bin/env python
import argparse
import csv
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import numpy as np

from runs.database import DataBase
from runs.database import RunEntry
from runs.logger import Logger
from scripts.crawl_events import crawl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patterns', nargs='*', type=PurePath)
    # parser.add_argument('--components', required=True, nargs='+')
    parser.add_argument('--tensorboard-dir', default=Path('.runs/tensorboard'), type=Path)
    parser.add_argument('--smoothing', type=int, default=2000)
    parser.add_argument('--tag', default='reward')
    parser.add_argument('--db-path', default='runs.db')
    parser.add_argument('--update-cache', action='store_true')
    args = parser.parse_args()

    header, rows = table(tag=args.tag,
                         db_path=args.db_path,
                         patterns=args.patterns,
                         smoothing=args.smoothing,
                         tensorboard_dir=args.tensorboard_dir,
                         use_cache=not args.update_cache)
    print(*header, sep=',')
    for row in rows:
        print(*row, sep=',')

    fig = plt.figure()
    # if len(args.components) == 1:
    #     projection = '2d'
    # elif len(args.components) == 2:
    #     projection = '3d'
    # else:
    #     raise RuntimeError('Must have )

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


def table(tag, db_path, patterns, smoothing, tensorboard_dir, use_cache):
    logger = Logger(quiet=False)
    with DataBase(db_path, logger) as db:
        entries = {entry.path: entry for entry in db.get(patterns)}
    dirs = [Path(tensorboard_dir, path) for path in entries]
    data_points = crawl(dirs=dirs, tag=tag,
                        smoothing=smoothing,
                        use_cache=use_cache)
    rewards = {str(event_file).replace(tensorboard_dir, ''): data
               for data, event_file in data_points}
    assert set(rewards.values()) == set(entries.values())

    header = list(RunEntry.fields()) + ['rewards']
    rows = [
        [getattr(entry, field) for field in RunEntry.fields()]
        + [rewards[path]]
        for path, entry in entries.items()
    ]
    return header, rows


if __name__ == '__main__':
    main()
