#! /usr/bin/env python

import argparse
from pathlib import Path

from sac.utils import collect_events_files, collect_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_names', nargs='*', type=Path)
    parser.add_argument('--tensorboard_dir', type=Path, default=Path('.runs/tensorboard'))
    parser.add_argument('--smoothing', type=int, default=2000)
    parser.add_argument('--reward-file', default='reward')
    args = parser.parse_args()
    event_files = collect_events_files(args.tensorboard_dir, args.run_names)
    for event_file_path in event_files:
        print(event_file_path)
        reward = collect_reward(event_file_path, args.smoothing)
        if reward:
            reward_path = Path(event_file_path.parent, args.reward_file)
            print(f'Writing {reward_path}...')
            with reward_path.open('w') as f:
                f.write(str(reward))
        else:
            print('No rewards found')


if __name__ == '__main__':
    main()
