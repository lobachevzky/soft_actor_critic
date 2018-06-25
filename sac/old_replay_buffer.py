import numpy as np

from sac import replay_buffer
from sac.array_group import ArrayGroup


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

    def sample(self, batch_size, comparison_buffer: replay_buffer.ReplayBuffer=None):
        top_pos = self.maxlen if self.full else self.pos
        indices = np.random.randint(0, top_pos, size=batch_size)
        assert isinstance(indices, np.ndarray)
        samples = []
        for idx in indices:
            sample = self.buffer[idx]
            samples.append(sample)
        ret_val = tuple(map(list, zip(*samples)))
        if comparison_buffer:
            assert ArrayGroup(ret_val) == ArrayGroup(comparison_buffer[indices])
        return ret_val

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def __getitem__(self, item):
        def get_item(index):
            return self.buffer[(self.pos + index) % self.maxlen]

        if isinstance(item, slice):
            return map(
                get_item,
                range(item.start or 0, item.stop
                      or (0 if item.start < 0 else self.maxlen), item.step or 1))
        else:
            try:
                return map(get_item, item)
            except TypeError:
                return get_item(item)
