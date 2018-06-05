from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = [None for _ in range(maxlen)]
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

    def extend(self, xs):
        for x in xs:
            self.append(x)

    def sample(self, batch_size):
        top_pos = self.maxlen if self.full else self.pos
        indices = np.random.randint(0, top_pos, size=batch_size)
        assert isinstance(indices, np.ndarray)
        samples = []
        for idx in indices:
            sample = self.buffer[idx]
            samples.append(sample)
        return tuple(map(list, zip(*samples)))

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def __getitem__(self, item):
        def get_item(index):
            return self.buffer[(self.pos + index) % self.maxlen]

        if isinstance(item, slice):
            return map(get_item,
                       range(item.start or 0,
                             item.stop or (0 if item.start < 0 else self.maxlen),
                             item.step or 1))
        else:
            try:
                return map(get_item, item)
            except TypeError:
                return get_item(item)
