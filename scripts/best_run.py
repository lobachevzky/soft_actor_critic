#! /usr/bin/env python

import argparse
from pathlib import Path

import re

from sac.utils import collect_events_files, collect_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='*', type=Path)
    parser.add_argument('--smoothing', type=int, default=2000)
    args = parser.parse_args()

    def reward(event_file):
        reward = collect_reward(event_file, args.smoothing)
        print(event_file, reward, sep=':\t')
        return reward or -float('inf')

    event_files = collect_events_files(args.dirs)
    if event_files:
        best_file = str(max(event_files, key=reward))
        print('Best run:')
        print(re.sub('.runs/tensorboard/|/events.out.tfevents.*', '', best_file))
    else:
        print('No event files found.')

if __name__ == '__main__':
    main()

