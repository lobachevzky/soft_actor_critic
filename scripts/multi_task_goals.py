# stdlib
import itertools

# third party
import click
import numpy as np
from gym import spaces


def parse_coordinate(ctx, param, string):
    if string is None:
        return
    return np.array(list(map(float, string.split(','))))


@click.command()
@click.option('--low', default='-.11,-.19,.412', callback=parse_coordinate)
@click.option('--high', default='.08,.19,.412', callback=parse_coordinate)
def cli(low, high):
    goal_space = spaces.Box(low=low, high=high)
    intervals = [2, 3, 1]
    x, y, z = [
        np.linspace(l, h, n)
        for l, h, n in zip(goal_space.low, goal_space.high, intervals)
    ]
    goals = np.array(list(itertools.product(x, y, z)))
    print('|'.join([','.join(map(str, np.round(g, 3))) for g in goals]))


if __name__ == '__main__':
    cli()
