#! /usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from pandas.errors import EmptyDataError

from sac.utils import softmax


def parse_double(string, delimiter='x'):
    if string is None:
        return
    a, b = map(float, string.split(delimiter))
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=Path)
    parser.add_argument('--components', nargs='+')
    parser.add_argument('--delimiter', default='|')
    parser.add_argument('--plot-path', default=None)
    parser.add_argument('--min-rows', type=int, default=15)
    parser.add_argument('--weight-key')
    parser.add_argument('--fig-size', type=parse_double)
    args = parser.parse_args()

    plot(labels=args.components,
         delimiter=args.delimiter,
         csv_dir=args.csv_dir,
         plot_path=args.plot_path,
         weight_key=args.weight_key,
         min_rows=args.min_rows)

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


def plot(csv_dir,
         delimiter,
         labels,
         plot_path,
         weight_key,
         min_rows):
    csv_paths = list(csv_dir.glob('*.csv'))
    x_label, y_label, z_label, *legend_labels = labels
    xyz_labels = [x_label, y_label, z_label]
    fig = plt.figure()
    if weight_key:
        nrows = ncols = 1
    else:
        nrows = int(len(csv_paths) ** (1/2))
        ncols = len(csv_paths) / nrows
    ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
    i = 1
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, delimiter=delimiter, index_col='path')
        except EmptyDataError:
            continue
        if df.shape[0] < min_rows:
            continue

        def filter_in_df(keys):
            return [k for k in keys if k in df]

        if weight_key:
            scatter_weighted(ax=ax, weight_key=weight_key,
                             df=df[filter_in_df(labels + [weight_key])],
                             labels=xyz_labels)

        else:
            scatter(ax=fig.add_subplot(nrows, ncols, i, projection='3d'),
                    df=df[filter_in_df(labels)],
                    legend_labels=filter_in_df(legend_labels),
                    xyz_labels=xyz_labels)
            i += 1

    plt.tight_layout()

    if plot_path is None:
        plt.show(aspect='auto')
    else:
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)


def format_label(k, v):
    if .01 < abs(v) < 1000:
        return '{}: {}'.format(k, v)
    return '{}: {:.1e}'.format(k, v)


def scatter(ax, df, legend_labels, xyz_labels):
    if legend_labels:
        for label_values, group in df.groupby(legend_labels):
            # if not isinstance(label_values, list):
            #     label = format_label(legend_labels[0], label_values)
            # else:
            try:
                label = ', '.join(format_label(k, v)
                                  for k, v in zip(legend_labels, label_values))
            except TypeError:
                label = format_label(legend_labels[0], label_values)

            ax.scatter(*group[xyz_labels].values.T, label=label)
    else:
        ax.scatter(*df[xyz_labels].values.T)
    ax.legend()


def scatter_weighted(ax, df, weight_key, labels):
    weights = softmax(df[weight_key])
    series = df.mul(weights, axis=0).sum()
    ax.scatter(*series[labels])


if __name__ == '__main__':
    main()
