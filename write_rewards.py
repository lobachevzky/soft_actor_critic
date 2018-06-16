#! /usr/bin/env python

import tensorflow as tf
from collections import deque
from pathlib import Path
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--smoothing', type=int, default=20)
    args = parser.parse_args()
    for path in args.path.glob('**/events*'):
        print(path.parent)
        q = deque(maxlen=args.smoothing)
        for event in tf.train.summary_iterator(str(path)):
            reward = next((v.simple_value for v in event.summary.value
                            if v.tag == 'reward'), None)
            if reward is not None:
                q += [reward]
        with Path(path.parent, 'reward').open('w') as f:
            f.write(str(sum(q) / float(len(q))))

if __name__ == '__main__':
    main()
