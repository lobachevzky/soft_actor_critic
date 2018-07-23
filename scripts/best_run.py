#! /usr/bin/env python

import argparse
from pathlib import Path

import re

from sac.utils import collect_events_files, collect_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_names', nargs='*', type=Path)
    parser.add_argument('--tensorboard_dir', type=Path, default=Path('.runs/tensorboard'))
    parser.add_argument('--smoothing', type=int, default=2000)
    args = parser.parse_args()

    def reward(event_file):
        return collect_reward(event_file, args.smoothing) or -float('inf')

    event_files = collect_events_files(args.tensorboard_dir, args.run_names)
    if event_files:
        best_file = str(max(event_files, key=reward))
        print(re.sub('.runs/tensorboard/|/events.out.tfevents.*', '', best_file))
    else:
        print('No event files found.')

if __name__ == '__main__':
    main()

