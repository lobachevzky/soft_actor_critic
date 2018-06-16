#! /usr/bin/env python

import tensorflow as tf
from collections import deque
from pathlib import Path

for run_dir in Path(".runs/tensorboard/5dof").iterdir():
    events_file = next((f for f in run_dir.iterdir()
                        if f.stem.startswith('events')), None)
    if events_file:
        q = deque(maxlen=20)
        for event in tf.train.summary_iterator(str(events_file)):
            reward = next((v.simple_value for v in event.summary.value
                           if v.tag == 'reward'), None)
            if reward is not None:
                q += [reward]
        with Path(run_dir, 'reward').open('w') as f:
            f.write(str(sum(q) / float(len(q))))
