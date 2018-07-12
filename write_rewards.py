#! /usr/bin/env python

import argparse
from itertools import islice
from pathlib import Path

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='*', type=Path, default=[Path('.runs/tensorboard')])
    parser.add_argument('--smoothing', type=int, default=20)
    parser.add_argument('--reward-file', default='reward')
    args = parser.parse_args()
    for path in (p for d in args.path for p in d.glob('**/events*')):
        print(path.parent)
        length = sum(1 for _ in tf.train.summary_iterator(str(path)))
        iterator = tf.train.summary_iterator(str(path))
        n_reward = args.smoothing
        events = islice(iterator, max(length - n_reward, 0), length)

        def get_reward(event):
            return next((v.simple_value for v in event.summary.value
                         if v.tag == args.reward_file), None)

        rewards = (get_reward(e) for e in events)
        rewards = [r for r in rewards if r is not None]
        if rewards:
            reward_path = Path(path.parent, args.reward_file)
            print(f'Writing {reward_path}...')
            with reward_path.open('w') as f:
                f.write(str(sum(rewards) / float(len(rewards))))
        else:
            print('No rewards found')


if __name__ == '__main__':
    main()
