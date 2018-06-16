#! /usr/bin/env python

import tensorflow as tf
from collections import deque
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=Path)
args = parser.parse_args()
for path in args.path.glob('**/events*'):
    print(path.parent)
    q = deque(maxlen=20)
    for event in tf.train.summary_iterator(str(path)):
        reward = next((v.simple_value for v in event.summary.value
                        if v.tag == 'reward'), None)
        if reward is not None:
            q += [reward]
    with Path(path.parent, 'reward').open('w') as f:
        f.write(str(sum(q) / float(len(q))))
