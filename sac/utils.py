from collections import namedtuple
from typing import Any, Optional, Union, Callable

import gym
import numpy as np
import tensorflow as tf


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def onehot(idx, num_entries):
    x = np.zeros(num_entries)
    x[idx] = 1
    return x


def horz_stack_images(*images, spacing=5, background_color=(0, 0, 0)):
    # assert that all shapes have the same siz
    if len(set([tuple(image.shape) for image in images])) != 1:
        raise Exception('All images must have same shape')
    if images[0].shape[2] != len(background_color):
        raise Exception('Depth of background color must be the same as depth of image.')
    height = images[0].shape[0]
    width = images[0].shape[1]
    depth = images[0].shape[2]
    canvas = np.ones([height, width * len(images) + spacing * (len(images) - 1), depth])
    bg_color = np.reshape(background_color, [1, 1, depth])
    canvas *= bg_color
    width_pos = 0
    for image in images:
        canvas[:, width_pos:width_pos + width, :] = image
        width_pos += (width + spacing)
    return canvas


def component(function):
    def wrapper(*args, **kwargs):
        reuse = kwargs.get('reuse', None)
        name = kwargs['name']
        if 'reuse' in kwargs:
            del kwargs['reuse']
        del kwargs['name']
        with tf.variable_scope(name, reuse=reuse):
            out = function(*args, **kwargs)
            variables = tf.get_variable_scope().get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)
            return out, variables

    return wrapper


def get_size(x):
    if x is None:
        return 0
    if np.isscalar(x):
        return 1
    return sum(map(get_size, x))


def assign_to_vector(x, vector: np.ndarray):
    try:
        dim = vector.size / vector.shape[-1]
    except ZeroDivisionError:
        return
    if np.isscalar(x):
        x = np.array([x])
    if isinstance(x, np.ndarray):
        vector.reshape(x.shape)[:] = x
    else:
        sizes = np.array(list(map(get_size, x)))
        sizes = np.cumsum(sizes / dim, dtype=int)
        for _x, start, stop in zip(x, [0] + list(sizes), sizes):
            indices = [slice(None) for _ in vector.shape]
            indices[-1] = slice(start, stop)
            assign_to_vector(_x, vector[tuple(indices)])


def vectorize(x, shape: Optional[tuple] = None):
    if isinstance(x, np.ndarray):
        return x

    size = get_size(x)
    vector = np.zeros(size)
    if shape:
        vector = vector.reshape(shape)

    assert isinstance(vector, np.ndarray)
    assign_to_vector(x=x, vector=vector)
    return vector


def normalize(vector: np.ndarray, low: np.ndarray, high: np.ndarray):
    low[np.isinf(low)] = -.5
    high[np.isinf(high)] = .5
    mean = (low + high) / 2
    dev = high - low
    dev[dev == 0] = 1
    return (vector - mean) / dev


def unwrap_env(env: gym.Env, condition: Callable[[gym.Env], bool]):
    while not condition(env):
        try:
            env = env.env
        except AttributeError:
            raise RuntimeError(f"env {env} has no children that meet condition.")
    return env


Obs = Any


class Step(namedtuple('Step', 's o1 a r o2 t')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


ArrayLike = Union[np.ndarray, list]
TRAIN_VALUES = """\
entropy
soft_update_xi_bar
V_loss
Q_loss
pi_loss
V_grad
Q_grad
pi_grad\
""".split('\n')
TrainStep = namedtuple('TrainStep', TRAIN_VALUES)
