# stdlib
from collections import namedtuple
from typing import Any, Callable, Sequence, Union

# third party
import gym
import numpy as np
import tensorflow as tf

Shape = Union[int, Sequence[int]]


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


def is_scalar(x):
    try:
        return np.shape(x) == ()
    except ValueError:
        return False


def get_size(x):
    if x is None:
        return 0
    if is_scalar(x):
        return 1
    return sum(map(get_size, x))


def assign_to_vector(x, vector: np.ndarray):
    try:
        dim = vector.size / vector.shape[-1]
    except ZeroDivisionError:
        return
    if is_scalar(x):
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


def vectorize(x, shape: Shape = None):
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


def get_space_attrs(space: gym.Space, attr: str):
    if hasattr(space, attr):
        return getattr(space, attr)
    elif isinstance(space, gym.spaces.Dict):
        return {k: get_space_attrs(v, attr) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        return [get_space_attrs(s, attr) for s in space.spaces]
    raise RuntimeError(f'{space} does not have attribute {attr}.')


def unwrap_env(env: gym.Env, condition: Callable[[gym.Env], bool]):
    while not condition(env):
        try:
            env = env.env
        except AttributeError:
            raise RuntimeError(f"env {env} has no children that meet condition.")
    return env


def create_sess():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.inter_op_parallelism_threads = 1
    return tf.Session(config=config)


def softmax(X, theta=1.0, axis=None):
    """
    Courtesy of https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    :param axis:
    """
    X = np.array(X)

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


Obs = Any


class Step(namedtuple('Step', 's o1 a r o2 t')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


ArrayLike = Union[np.ndarray, list]


def parse_double(ctx, param, string):
    if string is None:
        return
    a, b = map(float, string.split(','))
    return a, b
