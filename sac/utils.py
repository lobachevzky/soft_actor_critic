from collections import namedtuple
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Optional, Union, List

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
    mean = (low + high) / 2
    mean = np.clip(mean, -1e4, 1e4)
    mean[np.isnan(mean)] = 0
    dev = high - low
    dev[dev < 1e-3] = 1
    dev[np.isinf(dev)] = 1
    return (vector - mean) / dev


def unwrap_env(env: gym.Env, condition: Callable[[gym.Env], bool]):
    while not condition(env):
        try:
            env = env.env
        except AttributeError:
            raise RuntimeError(f"env {env} has no children that meet condition.")
    return env


def collect_reward(event_file_path: Path, n_rewards: int) -> Optional[float]:
    """
    :param event_file_path: path to events file
    :param n_rewards: number of rewards to average
    :return: average of last `n_rewards` in events file or None if events file is empty
    """
    length = sum(1 for _ in tf.train.summary_iterator(str(event_file_path)))
    iterator = tf.train.summary_iterator(str(event_file_path))
    events = islice(iterator, max(length - n_rewards, 0), length)

    def get_reward(event):
        return next((v.simple_value for v in event.summary.value
                     if v.tag == 'reward'), None)

    rewards = (get_reward(e) for e in events)
    rewards = [r for r in rewards if r is not None]
    try:
        return sum(rewards) / len(rewards)
    except ZeroDivisionError:
        return None


def collect_events_files(dirs):
    pattern = '**/events*'
    return [path for d in dirs for path in d.glob(pattern)]


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
