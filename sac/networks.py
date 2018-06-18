from collections import Iterable, Sequence, Callable

import numpy as np
import tensorflow as tf
from sac.agent import AbstractAgent
from sac.utils import Step, TrainStep, TRAIN_VALUES, ArrayLike
from tensorflow.contrib.rnn import LSTMBlockCell, LSTMStateTuple


def mlp(inputs, layer_size, n_layers, activation):
    for i in range(n_layers):
        inputs = tf.layers.dense(inputs, layer_size, activation, name='fc' + str(i))
    return inputs


class MlpNetwork(AbstractAgent):
    def network(self, inputs: tf.Tensor) -> tf.Tensor:
        return mlp(
            inputs=inputs,
            layer_size=self.layer_size,
            n_layers=self.n_layers,
            activation=self.activation)


class LstmNetwork(AbstractAgent):
    def __init__(self, batch_size: int, o_shape: Iterable, a_shape: Sequence, activation: Callable,
                 reward_scale: float, n_layers: int, layer_size: int, learning_rate: float,
                 grad_clip: float, device_num: int, num_lstm_units: int):
        with tf.device('/gpu:' + str(device_num)):
            state_shape = [batch_size] + list(layer_size)
            self.S = LSTMStateTuple(c=tf.placeholder(tf.float32, state_shape, name='C'),
                                    h=tf.placeholder(tf.float32, state_shape, name='H'))
            self.lstm = LSTMBlockCell(num_lstm_units)
            self.initial_state = self.lstm.zero_state(batch_size, tf.float32)
        super().__init__(batch_size=batch_size,
                         o_shape=o_shape,
                         a_shape=a_shape,
                         activation=activation,
                         reward_scale=reward_scale,
                         n_layers=n_layers,
                         layer_size=layer_size,
                         learning_rate=learning_rate,
                         grad_clip=grad_clip,
                         device_num=device_num)
        self.new_s = self.sess.run(self.initial_state)

    def network(self, inputs: tf.Tensor) -> tf.Tensor:
        output, self.new_s = self.lstm(inputs, self.S)
        return output

    def state_feed(self):
        return {self.S.c: self.new_s.c, self.S.h: self.new_s.h}

    def train_step(self, step: Step, feed_dict: dict = None) -> TrainStep:
        if feed_dict is None:
            feed_dict = {**self.state_feed(),
                         **{self.O1: step.s1,
                            self.A: step.a,
                            self.R: np.array(step.r) * self.reward_scale,
                            self.O2: step.s2,
                            self.T: step.t}
                         }
        return super().train_step(step, feed_dict)

    def get_actions(self, s1: ArrayLike, sample: bool = True) -> np.ndarray:
        feed_dict = {**{self.O1: s1}, **self.state_feed()}
        A = self.A_sampled1 if sample else self.A_max_likelihood
        actions, self.new_s = self.sess.run(A, feed_dict)
        return actions[0]
