from collections import deque, Iterable

import numpy as np

from sac.utils import Step


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = np.empty(maxlen, dtype=Step)
        self.pos = 0
        self.full = False
        self.empty = True

    def append(self, x):
        self.empty = False
        self.buffer[self.pos] = x
        self.pos += 1
        if self.pos >= self.maxlen:
            self.full = True
            self.pos = 0

    def extend(self, xs: Iterable):
        for x in xs:
            self.append(x)

    def sample(self, batch_size: int):
        top_pos = self.maxlen if self.full else self.pos
        indices = np.random.randint(0, top_pos, size=batch_size)  # type: np.ndarray
        samples = []
        for idx in indices:
            sample = self.buffer[idx]
            samples.append(sample)
        return tuple(map(list, zip(*samples)))

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def __getitem__(self, item):
        if isinstance(item, slice):
            return (self.buffer[i] for i in
                    range(item.start or 0, item.stop or len(self), item.step or 1))
        else:
            return self.buffer[item]
