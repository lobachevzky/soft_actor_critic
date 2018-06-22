#! /usr/bin/env python
from typing import Union, Iterator

import numpy as np

pos = 0
top_pos = 5


def initialize(x: Union[tuple, list, np.ndarray, int, float], maxlen: int):
    assert isinstance(x, (tuple, list, np.ndarray, int, float))
    a = np.array(x)
    print('a')
    print(a)
    if a.dtype in [float, int]:
        return np.zeros(a.shape)
    if a.dtype == object:
        return [initialize(_x, maxlen) for _x in x]
    else:
        raise RuntimeError('a is weird:', a)


def sample(batch_size, seq_len, buffer):
    indices = np.random.randint(0, top_pos, size=batch_size)  # type: np.ndarray
    if seq_len is not None:
        indices = np.array([np.arange(i, i + seq_len) for i in indices])
    if isinstance(buffer, np.ndarray):
        return buffer[indices]
    return [sample(batch_size, seq_len, q) for q in buffer]


def append(x: Union[tuple, list, np.ndarray, int, float],
           buffer: Union[list, np.ndarray]):
    assert isinstance(x, (tuple, list, np.ndarray, int, float))
    a = np.array(x)
    if a.dtype in [float, int]:
        assert isinstance(buffer, np.ndarray)
        if a.shape == ():
            buffer[pos] = x
        else:
            buffer[pos, :] = x
    if a.dtype == object:
        assert isinstance(x, Iterator)
        for _x, _buffer in zip(x, buffer):
            append(_x, _buffer)
    else:
        raise RuntimeError('a is weird:', a)


x = (np.arange(2), (np.arange(3), np.arange(4)))
for buffer, array in match_arrays(x, buffers):
    buffer[:] = array
