from typing import Tuple
from typing import Union, Iterable

import numpy as np


def homogeneous(x):
    return isinstance(x, (np.ndarray, float, int))


BufferInput = Union[tuple, list, np.ndarray, int, float]
Key = Union[int, slice, np.ndarray]


def initialize(x: BufferInput, maxlen: int):
    if homogeneous(x):
        shape = np.shape(x)  # type: Tuple[int]
        return np.zeros((maxlen,) + shape)
    else:
        assert isinstance(x, Iterable)
        return [initialize(_x, maxlen) for _x in x]


def getitem(buffer, key: np.ndarray):
    if homogeneous(buffer):
        return buffer[key]
    return [getitem(b, key) for b in buffer]


def setitem(buffer: Union[list, np.ndarray],
            key: Key, x: BufferInput):
    if homogeneous(x):
        assert isinstance(buffer, np.ndarray)
        if np.shape(x) == ():
            buffer[key] = x
        else:
            buffer[key, :] = x
    else:
        assert isinstance(x, Iterable)
        for _buffer, _x in zip(buffer, x):
            setitem(_buffer, key, _x)


class Buffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = None
        self.full = False
        self.pos = 0

    def __getitem__(self, indices: np.ndarray):
        assert self.buffer is not None
        return getitem(buffer=self.buffer, key=indices)

    def __setitem__(self, key, value):
        setitem(self.buffer, key=key, x=value)

    def __len__(self):
        return self.maxlen if self.full else self.pos

    def sample(self, batch_size, seq_len=None):
        top_pos = self.maxlen if self.full else self.pos
        indices = np.random.randint(0, top_pos, size=batch_size)  # type: np.ndarray
        if seq_len is not None:
            indices = np.array([np.arange(i, i + seq_len) for i in indices])
        assert isinstance(indices, np.ndarray)
        return self[indices]

    def append(self, x: BufferInput):
        if self.pos >= self.maxlen:
            self.full = True
            self.pos = 0

        if self.buffer is None:
            self.buffer = initialize(x=x, maxlen=self.maxlen)
        self[self.pos] = x
        self.pos += 1
        if self.pos == self.maxlen:
            self.pos = 0
            self.full = True

    def extend(self, xs: Iterable[BufferInput]):
        for x in xs:
            self.append(x)
