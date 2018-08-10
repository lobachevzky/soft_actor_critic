#! /usr/bin/env python

import argparse
import re
from pathlib import Path

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
        sorted_files = sorted([(get_reward(f), f) for f in event_files])
        print('\nEvents files, sorted worst to best:')
        for reward, event_file in sorted_files:
            print(
                re.sub('.runs/tensorboard/|/events.out.tfevents.*', '', str(event_file)),
                reward,
                sep='\t'
            )
    else:
        print('No event files found.')


if __name__ == '__main__':
    main()
