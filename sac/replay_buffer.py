import numpy as np

from sac.array_group import X, Key, ArrayGroup, get_shapes


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = None
        self.full = False
        self.pos = 0

    @property
    def empty(self):
        return self.buffer is None

    def __getitem__(self, key: Key):
        assert self.buffer is not None
        return self.buffer[self.modulate(key)]

    def __setitem__(self, key, value):
        self.buffer[self.modulate(key)] = value

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def modulate(self, key: Key):
        if isinstance(key, slice):
            key = np.arange(key.start or 0,
                            0 if key.stop is None else key.stop,
                            key.step)
        return (key + self.pos) % self.maxlen

    def sample(self, batch_size: int, seq_len=None):
        indices = np.random.randint(-len(self), 0, size=batch_size)  # type: np.ndarray
        if seq_len is not None:
            indices = np.array([np.arange(i, i + seq_len) for i in indices])
        assert isinstance(indices, np.ndarray)
        return self[indices]

    def append(self, x: X):
        if self.pos >= self.maxlen:
            self.full = True

        if self.buffer is None:
            self.buffer = ArrayGroup.shape_like(
                x=x, pre_shape=(self.maxlen,))
        n = common_dim(x)
        self[get_shapes(x)] = x
        self.pos = self.modulate(n or 1)
        if self.pos == self.maxlen:
            self.pos = 0
            self.full = True
