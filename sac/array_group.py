from collections import namedtuple
from numbers import Number
from typing import Union, Iterable

import numpy as np

Input = Union[(tuple, list, np.ndarray, Number, bool)]
Key = Union[int, slice, np.ndarray]


def getitem(array_group, key: np.ndarray):
    if isinstance(array_group, np.ndarray):
        return array_group[key]
    return [getitem(a, key) for a in array_group]


def setitem(array_group: Union[list, np.ndarray],
            key: Key, x: Input):
    if isinstance(array_group, np.ndarray):
        if np.shape(x) == ():
            array_group[key] = x
        else:
            array_group[key, :] = x
    else:
        assert isinstance(x, Iterable)
        for _group, _x in zip(array_group, x):
            setitem(_group, key, _x)


def allocate(pre_shape: tuple, shapes: Union[tuple, Iterable]):
    try:
        return np.zeros(pre_shape + shapes)
    except TypeError:
        return [allocate(pre_shape, shape) for shape in shapes]


def get_shapes(x, subset=None):
    if isinstance(x, np.ndarray):
        shape = np.shape(x)  # type: tuple
        if subset is None:
            return shape
        return shape[subset]
    if np.isscalar(x):
        return tuple()
    return [get_shapes(_x, subset) for _x in x]


class ArrayGroup:
    @staticmethod
    def shape_like(x: Input, pre_shape: tuple):
        return ArrayGroup(allocate(pre_shape=pre_shape,
                                   shapes=(get_shapes(x))))

    def __init__(self, values):
        self.arrays = values

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, key: Key):
        return ArrayGroup(getitem(self.arrays, key=key))

    def __setitem__(self, key: Key, value):
        setitem(self.arrays, key=key, x=value)
