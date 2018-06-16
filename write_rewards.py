#! /usr/bin/env python

import tensorflow as tf
from collections import deque
from pathlib import Path
import sys
import argparse
from itertools import islice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='*', type=Path, default=[Path('.runs/tensorboard')])
    parser.add_argument('--smoothing', type=int, default=20)
    args = parser.parse_args()
    for path in (p for d in args.path for p in d.glob('**/events*')):
        print(path.parent)
        length = sum(1 for _ in tf.train.summary_iterator(str(path)))
        iterator = tf.train.summary_iterator(str(path))
        n_reward = args.smoothing
        events = islice(iterator, length - n_reward, length)

        def get_reward(event):
            return next((v.simple_value for v in event.summary.value
                         if v.tag == 'reward'), None)

        rewards = (get_reward(e) for e in events)
        rewards = [r for r in rewards if r is not None]
        with Path(path.parent, 'reward').open('w') as f:
            f.write(str(sum(rewards) / float(len(rewards))))

if __name__ == '__main__':
    main()
