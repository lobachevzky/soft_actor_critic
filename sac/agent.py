from abc import abstractmethod
from collections import namedtuple
from typing import Callable, Iterable, Sequence, List

import numpy as np
import tensorflow as tf

from sac.utils import Step, TrainStep, TRAIN_VALUES, ArrayLike

NetworkOutput = namedtuple('NetworkOutput', 'output state')


class AbstractAgent:
    def __init__(self, sess: tf.Session, batch_size: int, seq_len: int,
                 o_shape: Iterable, a_shape: Sequence,
                 activation: Callable, reward_scale: float, n_layers: int, layer_size: int,
                 learning_rate: float, grad_clip: float, device_num: int,
                 reuse=False) -> None:

        self.activation = activation
        self.n_layers = n_layers
        self.layer_size = layer_size
        self._seq_len = seq_len
        self.reward_scale = reward_scale
        self.initial_state = None
        self.sess = sess

        with tf.device('/gpu:' + str(device_num)), tf.variable_scope('agent', reuse=reuse):
            seq_dim = [batch_size]
            if self.seq_len is not None:
                seq_dim = [batch_size, self.seq_len]

            self.O1 = tf.placeholder(tf.float32, seq_dim + list(o_shape), name='O1')
            self.O2 = tf.placeholder(tf.float32, seq_dim + list(o_shape), name='O2')
            self.A = A = tf.placeholder(tf.float32, [batch_size] + list(a_shape), name='A')
            self.R = R = tf.placeholder(tf.float32, [batch_size], name='R')
            self.T = T = tf.placeholder(tf.float32, [batch_size], name='T')
            gamma = 0.99
            tau = 0.01

            with tf.variable_scope('pi'):
                processed_s, self.S_new = self.network(self.O1)
                parameters = self.parameters = self.produce_policy_parameters(
                    a_shape[0], processed_s)

            def pi_network_log_prob(a: tf.Tensor, name: str, reuse: bool) \
                    -> tf.Tensor:
                with tf.variable_scope(name, reuse=reuse):
                    return self.policy_parameters_to_log_prob(a, parameters)

            def sample_pi_network(name: str, reuse: bool) -> tf.Tensor:
                with tf.variable_scope(name, reuse=reuse):
                    return self.policy_parameters_to_sample(parameters)

            # generate actions:
            self.A_max_likelihood = tf.stop_gradient(
                self.policy_parameters_to_max_likelihood_action(parameters))
            self.A_sampled1 = A_sampled1 = tf.stop_gradient(
                sample_pi_network('pi', reuse=True))

            # constructing V loss
            with tf.control_dependencies([self.A_sampled1]):
                v1 = self.v_network(self.O1, 'V')
                q1 = self.q_network(self.O1, self.transform_action_sample(A_sampled1), 'Q')
                log_pi_sampled1 = pi_network_log_prob(A_sampled1, 'pi', reuse=True)
                self.V_loss = V_loss = tf.reduce_mean(
                    0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

            # constructing Q loss
            with tf.control_dependencies([self.V_loss]):
                v2 = self.v_network(self.O2, 'V_bar')
                q = self.q_network(
                    self.O1, self.transform_action_sample(A), 'Q', reuse=True)
                # noinspection PyTypeChecker
                self.Q_loss = Q_loss = tf.reduce_mean(
                    0.5 * tf.square(q - (R + (1 - T) * gamma * v2)))

            # constructing pi loss
            with tf.control_dependencies([self.Q_loss]):
                self.A_sampled2 = A_sampled2 = tf.stop_gradient(
                    sample_pi_network('pi', reuse=True))
                q2 = self.q_network(self.O1, self.transform_action_sample(A_sampled2), 'Q', reuse=True)
                log_pi_sampled2 = pi_network_log_prob(A_sampled2, 'pi', reuse=True)
                self.pi_loss = pi_loss = tf.reduce_mean(
                    log_pi_sampled2 * tf.stop_gradient(log_pi_sampled2 - q2 + v1))

            # grabbing all the relevant variables
            def get_variables(name: str) -> List[tf.Variable]:
                return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=f'agent/{name}/')
            phi, theta, xi, xi_bar = map(get_variables, ['pi', 'Q', 'V', 'V_bar'])

            def train_op(loss, var_list, dependency):
                with tf.control_dependencies([dependency]):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    gradients, variables = zip(
                        *optimizer.compute_gradients(loss, var_list=var_list))
                    if grad_clip:
                        gradients, norm = tf.clip_by_global_norm(gradients, grad_clip)
                    else:
                        norm = tf.global_norm(gradients)
                    op = optimizer.apply_gradients(zip(gradients, variables))
                    return op, norm

            self.train_V, self.V_grad = train_op(
                loss=V_loss, var_list=xi, dependency=self.pi_loss)
            self.train_Q, self.Q_grad = train_op(
                loss=Q_loss, var_list=theta, dependency=self.train_V)
            self.train_pi, self.pi_grad = train_op(
                loss=pi_loss, var_list=phi, dependency=self.train_Q)

            with tf.control_dependencies([self.train_pi]):
                soft_update_xi_bar_ops = [
                    tf.assign(xbar, tau * x + (1 - tau) * xbar)
                    for (xbar, x) in zip(xi_bar, xi)
                    ]
                self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
                # self.check = tf.add_check_numerics_ops()
                self.entropy = tf.reduce_mean(self.entropy_from_params(self.parameters))
                # ensure that xi and xi_bar are the same at initialization

            sess.run(tf.global_variables_initializer())

            # ensure that xi and xi_bar are the same at initialization
            hard_update_xi_bar_ops = [tf.assign(xbar, x) for (xbar, x) in zip(xi_bar, xi)]

            hard_update_xi_bar = tf.group(*hard_update_xi_bar_ops)
            sess.run(hard_update_xi_bar)
            self.saver = tf.train.Saver()

    @property
    def seq_len(self):
        return self._seq_len

    def train_step(self, step: Step, feed_dict: dict = None) -> TrainStep:
        if feed_dict is None:
            feed_dict = {
                self.O1: step.o1,
                self.A: step.a,
                self.R: np.array(step.r) * self.reward_scale,
                self.O2: step.o2,
                self.T: step.t
            }
        return TrainStep(*self.sess.run([getattr(self, attr)
                                         for attr in TRAIN_VALUES], feed_dict))

    def get_actions(self, o: ArrayLike, _, sample: bool = True) -> NetworkOutput:
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return NetworkOutput(output=self.sess.run(A, {self.O1: [o]})[0],
                             state=0)

    def q_network(self, s: tf.Tensor, a: tf.Tensor, name: str,
                  reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            s = self.network(s).output
            sa = tf.concat([s, a], axis=1)
            return tf.reshape(tf.layers.dense(sa, 1, name='q'), [-1])

    def v_network(self, s: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.network(s).output, 1, name='v'), [-1])

    @abstractmethod
    def network(self, inputs: tf.Tensor) -> NetworkOutput:
        pass

    @abstractmethod
    def produce_policy_parameters(self, a_shape: Iterable,
                                  processed_o: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_log_prob(self, a: tf.Tensor,
                                      parameters: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_sample(self, parameters: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def policy_parameters_to_max_likelihood_action(self, parameters) -> tf.Tensor:
        pass

    @abstractmethod
    def transform_action_sample(self, action_sample: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def entropy_from_params(self, params: tf.Tensor) -> tf.Tensor:
        pass
