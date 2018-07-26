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

    def get_reward(event_file):
        reward = collect_reward(event_file, args.smoothing)
        print('file:', event_file, '\treward:', reward)
        return reward or -float('inf')

    event_files = collect_events_files(args.dirs)
    if event_files:
        sorted_files = sorted(event_files, key=get_reward)
        print('\nEvents files, sorted worst to best:')
        for event_file in sorted_files:
            print(re.sub('.runs/tensorboard/|/events.out.tfevents.*', '', str(event_file)))
    else:
        print('No event files found.')

if __name__ == '__main__':
    main()

