from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple

from sac.agent import AbstractAgent, NetworkOutput
from sac.utils import ArrayLike, Step, TrainStep
from collections import namedtuple


def mlp(inputs, layer_size, n_layers, activation):
    for i in range(n_layers):
        inputs = tf.layers.dense(inputs, layer_size, activation, name='fc' + str(i))
    return inputs


class MlpAgent(AbstractAgent):
    @property
    def seq_len(self):
        return None

    def network(self, inputs: tf.Tensor) -> NetworkOutput:
        return NetworkOutput(
            output=mlp(
                inputs=inputs,
                layer_size=self.layer_size,
                n_layers=self.n_layers,
                activation=self.activation),
            state=None)


class SACXAgent(AbstractAgent):
    @property
    def seq_len(self):
        return None

    @staticmethod
    def actor_network(inputs: tf.Tensor) -> NetworkOutput:
        l1 = tf.layers.dense(inputs, 200, tf.nn.elu)
        l2 = tf.contrib.layers.layer_norm(l1)
        l3 = tf.layers.dense(l2, 200, tf.nn.elu)
        return NetworkOutput(output=tf.layers.dense(l3, 100, tf.nn.elu), state=None)

    @staticmethod
    def critic_network(inputs: tf.Tensor) -> NetworkOutput:
        l1 = tf.layers.dense(inputs, 400, tf.nn.elu)
        l2 = tf.contrib.layers.layer_norm(l1)
        l3 = tf.layers.dense(l2, 400, tf.nn.elu)
        return NetworkOutput(output=tf.layers.dense(l3, 200, tf.nn.elu), state=None)

    def pi_network(self, o: tf.Tensor):
        with tf.variable_scope('pi'):
            return self.actor_network(o)

    def q_network(self, o: tf.Tensor, a: tf.Tensor, name: str,
                  reuse: bool = None):
        with tf.variable_scope(name, reuse=reuse):
            oa = tf.concat([o, a], axis=1)
            return tf.reshape(tf.layers.dense(self.critic_network(oa).output, 1, name='q'), [-1])

    def v_network(self, o: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.critic_network(o).output, 1, name='v'), [-1])


class LstmAgent(AbstractAgent):
    @property
    def seq_len(self):
        if self._seq_len:
            return self._seq_len
        return 8

    def __init__(self, batch_size: int, layer_size: int, device_num: int, **kwargs):
        self.batch_size = batch_size
        self.layer_size = layer_size
        with tf.device('/gpu:' + str(device_num)):
            state_args = tf.float32, [batch_size, layer_size]
            self.S = LSTMStateTuple(
                c=tf.placeholder(*state_args, name='C'),
                h=tf.placeholder(*state_args, name='H'))
            self.lstm = BasicLSTMCell(layer_size)
        super().__init__(
            batch_size=batch_size, layer_size=layer_size, device_num=device_num, **kwargs)
        self.initial_state = self.sess.run(self.lstm.zero_state(batch_size, tf.float32))
        assert np.shape(self.initial_state) == (2, batch_size, layer_size)
        assert self.S.c.shape == self.S.h.shape == (batch_size, layer_size)

    def network(self, inputs: tf.Tensor, reuse=False) -> tf.Tensor:
        split_inputs = tf.split(inputs, self.seq_len, axis=1)
        s = self.S
        for x in split_inputs:
            x = tf.squeeze(x, axis=1)
            outputs = NetworkOutput(*self.lstm(x, s))
        return outputs

    def state_feed(self, states):
        return dict(zip(self.S, states))

    def train_step(self, step: Step, feed_dict: dict = None) -> TrainStep:
        assert np.shape(step.s) == np.shape(self.initial_state)
        if feed_dict is None:
            feed_dict = {
                **self.state_feed(step.s),
                **{
                    self.O1: step.o1,
                    self.A: step.a,
                    self.R: np.array(step.r) * self.reward_scale,
                    self.O2: step.o2,
                    self.T: step.t
                }
            }
        return super().train_step(step, feed_dict)

    def q_network(self, o: tf.Tensor, a: tf.Tensor, name: str, reuse: bool = None) \
            -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            o = self.network(o).output
            oa = tf.concat([o, a], axis=1)
            return tf.reshape(tf.layers.dense(oa, 1, name='q'), [-1])

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) \
            -> Tuple[np.ndarray, LSTMStateTuple]:
        assert len(np.shape(o)) == 1
        assert np.shape(state) == np.shape(self.initial_state)
        feed_dict = {**{self.O1: [[o]]}, **self.state_feed(state)}
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return self.sess.run([A[0], self.S_new], feed_dict)


class MoEAgent(AbstractAgent):
    def __init__(self, n_networks, device_num, **kwargs):
        self.n_networks = n_networks
        super().__init__(device_num=device_num, **kwargs)

    @property
    def seq_len(self):
        return None

    def network(self, inputs: tf.Tensor):
        with tf.variable_scope('weights'):
            weights = mlp(
                inputs,
                self.layer_size,
                n_layers=self.n_layers - 1,
                activation=self.activation)
            weights = tf.layers.dense(weights, units=self.n_networks)
            weights = tf.nn.softmax(logits=weights, axis=-1)
        weights = tf.expand_dims(weights, axis=1)  # [batch, hidden, networks]

        def vote(i):
            with tf.variable_scope('vote' + str(i)):
                return mlp(inputs, self.layer_size, self.n_layers - 1, self.activation)

        h = tf.stack(
            values=list(map(vote, range(self.n_networks))),
            axis=-1)  # [batch, hidden, networks]
        h = tf.reduce_sum(h * weights, axis=2)
        output = tf.layers.dense(h, units=self.layer_size, name='output')
        return NetworkOutput(output=output, state=weights)


VariableTypes = namedtuple('VariableTypes', 'pi Q V')


class GradClusterAgent(AbstractAgent):
    def __init__(self, n_networks, device_num, **kwargs):
        self.n_networks = n_networks
        super().__init__(device_num=device_num, **kwargs)
        expert_gradients = self.S
        grads = VariableTypes(pi=self.pi_grad,
                              V=self.V_grad,
                              Q=self.Q_grad)
        expert_scopes = VariableTypes(pi='pi_expert',
                                      V='V_expert',
                                      Q='Q_expert')
        weight_scopes = VariableTypes(pi='pi_weight',
                                      V='V_weight',
                                      Q='Q_weight')
        inputs = VariableTypes(
            pi=self.O1,
            V=self.O1,
            Q=self.Q_input(self.transform_action_sample(self.A_sampled2)),
        )

        with tf.variable_scope('weights'):
            self.inferred_weights = VariableTypes(*[
                self.inferred_weight(*args)
                for args in zip(inputs, weight_scopes)
            ])
        best_experts = VariableTypes(*[
            self.best_expert(*args)
            for args in zip(grads, expert_gradients, expert_scopes)
        ])
        self.weights_loss = sum([
            .5 * (inferred_weight - tf.one_hot(best_expert)) ** 2
            for inferred_weight, best_expert
            in zip(self.inferred_weights, best_experts)
        ])
        weights_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'agent/weights/')
        self.train_weights, _, self.weights_grad_norm = self.train_op(
            loss=self.weights_loss, var_list=weights_variables)
        self.sess.run(tf.global_variables_initializer())
        self.initial_state = [0, 0, 0]
        self.S_new = VariableTypes(*[.99 * old + .01 * new for old, new in zip(self.S, grads)])

    def Q_input(self, actions):
        return tf.concat([self.O1, actions], axis=1)

    @staticmethod
    def best_expert(gradient, expert_gradients, scope):
        with tf.variable_scope(scope):
            return tf.reduce_min([tf.global_norm(gradient - expert_gradient)
                                  for expert_gradient in expert_gradients])

    def inferred_weight(self, inputs, scope):
        with tf.variable_scope(scope):
            inferred_weights = mlp(
                inputs,
                self.layer_size,
                n_layers=self.n_layers - 1,
                activation=self.activation)
            inferred_weights = tf.layers.dense(inferred_weights, units=self.n_networks)
            return tf.nn.softmax(logits=inferred_weights, axis=-1)

    def state_feed(self, states):
        return dict(zip(self.S, states))

    def train_step(self, step: Step, feed_dict: dict = None) -> TrainStep:
        assert np.shape(step.s) == np.shape(self.initial_state)
        if feed_dict is None:
            feed_dict = {
                **self.state_feed(step.s),
                **{
                    self.O1: step.o1,
                    self.A: step.a,
                    self.R: np.array(step.r) * self.reward_scale,
                    self.O2: step.o2,
                    self.T: step.t
                }
            }
        return super().train_step(step, feed_dict)

    @property
    def seq_len(self):
        return None

    def pi_network(self, o: tf.Tensor):
        with tf.variable_scope('pi'):
            return self._network(o, self.inferred_weights.pi)

    def q_network(self, o: tf.Tensor, a: tf.Tensor, name: str,
                  reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            Q = self._network(self.Q_input(a), self.inferred_weights.Q).output
            Q = tf.layers.dense(Q, 1, name='q')
            return tf.reshape(Q, [-1])

    def v_network(self, o: tf.Tensor, name: str, reuse: bool = None):
        with tf.variable_scope(name, reuse=reuse):
            V = self._network(o, self.inferred_weights.V).output
            return tf.reshape(tf.layers.dense(V, 1, name='v'), [-1])

    def _network(self, inputs: tf.Tensor, weights: tf.Tensor):
        def expert(i):
            with tf.variable_scope('expert' + str(i)):
                return mlp(inputs, self.layer_size, self.n_layers - 1, self.activation)

        h = tf.stack(
            values=list(map(expert, range(self.n_networks))),
            axis=-1)  # [batch, hidden, networks]
        h = tf.reduce_sum(h * weights, axis=2)
        output = tf.layers.dense(h, units=self.layer_size, name='output')
        return NetworkOutput(output=output, state=weights)
