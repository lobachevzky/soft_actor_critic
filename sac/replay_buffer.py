import numpy as np



class ReplayBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffers = None
        self.pos = 0
        self.full = False
        self.empty = True

    def initialize(self, x):
        dims =

    def append(self, *xs):
        assert len(xs) == len(self.buffers)
        self.empty = False
        for buffer, x in zip(self.buffers, xs):
            buffer[self.pos, :] = x
        self.pos += 1
        if self.pos >= self.maxlen:
            self.full = True
            self.pos = 0

    def extend(self, xs):
        for x in xs:
            self.append(*x)

    def sample(self, batch_size, seq_len=None):
        top_pos = self.maxlen if self.full else self.pos
        indices = np.random.randint(0, top_pos, size=batch_size)  # type: np.ndarray
        if seq_len is not None:
            indices = np.array([np.arange(i, i + seq_len) for i in indices])
        assert isinstance(indices, np.ndarray)
        return [buffer[indices] for buffer in self.buffers]

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def __getitem__(self, item):
        def get_item(index):
            return self.buffers[(self.pos + index) % self.maxlen]

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
