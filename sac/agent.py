# stdlib
from abc import abstractmethod
from collections import namedtuple
from typing import Callable, Iterable, List, Sequence

# third party
import numpy as np
import tensorflow as tf

# first party
from sac.utils import ArrayLike, Step

NetworkOutput = namedtuple('NetworkOutput', 'output state')


class AbstractAgent:
    def __init__(self,
                 sess: tf.Session,
                 batch_size: int,
                 seq_len: int,
                 o_shape: Iterable,
                 a_shape: Sequence,
                 activation: Callable,
                 reward_scale: float,
                 entropy_scale: float,
                 n_layers: int,
                 layer_size: int,
                 learning_rate: float,
                 grad_clip: float,
                 device_num: int = 1,
                 reuse=False,
                 scope='agent') -> None:

        self.default_train_values = [
            'entropy',
            'soft_update_xi_bar',
            'V_loss',
            'Q_loss',
            'pi_loss',
            'V_grad',
            'Q_grad',
            'pi_grad',
            'train_V',
            'train_Q',
            'train_pi',
        ]
        self.reward_scale = reward_scale
        self.activation = activation
        self.n_layers = n_layers
        self.layer_size = layer_size
        self._seq_len = seq_len
        self.initial_state = None
        self.sess = sess

        with tf.device('/gpu:' + str(device_num)), tf.variable_scope(scope, reuse=reuse):
            self.global_step = tf.Variable(0, name='global_step')

            seq_dim = [batch_size]
            if self.seq_len is not None:
                seq_dim = [batch_size, self.seq_len]

            self.O1 = tf.placeholder(tf.float32, seq_dim + list(o_shape), name='O1')
            self.O2 = tf.placeholder(tf.float32, seq_dim + list(o_shape), name='O2')
            self.A = A = tf.placeholder(
                tf.float32, [batch_size] + list(a_shape), name='A')
            self.R = R = tf.placeholder(tf.float32, [batch_size], name='R')
            self.T = T = tf.placeholder(tf.float32, [batch_size], name='T')
            gamma = tf.constant(0.99)
            tau = 0.01

            processed_s, self.S_new = self.pi_network(self.O1)
            parameters = self.parameters = self.produce_policy_parameters(
                a_shape[0], processed_s)

            def pi_network_log_prob(a: tf.Tensor, name: str, _reuse: bool) \
                    -> tf.Tensor:
                with tf.variable_scope(name, reuse=_reuse):
                    return self.policy_parameters_to_log_prob(a, parameters)

            def sample_pi_network(name: str, _reuse: bool) -> tf.Tensor:
                with tf.variable_scope(name, reuse=_reuse):
                    return self.policy_parameters_to_sample(parameters)

            # generate actions:
            self.A_max_likelihood = tf.stop_gradient(
                self.policy_parameters_to_max_likelihood_action(parameters))
            self.A_sampled1 = A_sampled1 = tf.stop_gradient(
                sample_pi_network('pi', _reuse=True))

            # constructing V loss
            v1 = self.v_network(self.O1, 'V')
            self.v1 = v1
            q1 = self.q_network(self.O1, self.transform_action_sample(A_sampled1), 'Q')
            log_pi_sampled1 = pi_network_log_prob(A_sampled1, 'pi', _reuse=True)
            log_pi_sampled1 *= entropy_scale  # type: tf.Tensor
            self.V_loss = V_loss = tf.reduce_mean(
                0.5 * tf.square(v1 - (q1 - log_pi_sampled1)))

            # constructing Q loss
            v2 = self.v_network(self.O2, 'V_bar')
            q = self.q_network(self.O1, self.transform_action_sample(A), 'Q', reuse=True)
            not_done = 1 - T  # type: tf.Tensor
            q_target = R + gamma * not_done * v2
            self._td_error = tf.square(q - q_target)
            self.Q_loss = Q_loss = tf.reduce_mean(0.5 * self._td_error)

            # constructing pi loss
            self.A_sampled2 = A_sampled2 = tf.stop_gradient(
                sample_pi_network('pi', _reuse=True))
            q2 = self.q_network(
                self.O1, self.transform_action_sample(A_sampled2), 'Q', reuse=True)
            log_pi_sampled2 = pi_network_log_prob(A_sampled2, 'pi', _reuse=True)
            log_pi_sampled2 *= entropy_scale  # type: tf.Tensor
            self.pi_loss = pi_loss = tf.reduce_mean(
                log_pi_sampled2 * tf.stop_gradient(log_pi_sampled2 - q2 + v1))

            # grabbing all the relevant variables
            def get_variables(var_name: str) -> List[tf.Variable]:
                return tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{scope}/{var_name}/')

            phi, theta, xi, xi_bar = map(get_variables, ['pi', 'Q', 'V', 'V_bar'])

            def train_op(loss, var_list):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                gradients, variables = zip(
                    *optimizer.compute_gradients(loss, var_list=var_list))
                if grad_clip:
                    gradients, norm = tf.clip_by_global_norm(gradients, grad_clip)
                else:
                    norm = tf.global_norm(gradients)
                op = optimizer.apply_gradients(
                    zip(gradients, variables), global_step=self.global_step)
                return op, norm

            self.train_V, self.V_grad = train_op(loss=V_loss, var_list=xi)
            self.train_Q, self.Q_grad = train_op(loss=Q_loss, var_list=theta)
            self.train_pi, self.pi_grad = train_op(loss=pi_loss, var_list=phi)

            soft_update_xi_bar_ops = [
                tf.assign(xbar, tau * x + (1 - tau) * xbar)
                for (xbar, x) in zip(xi_bar, xi)
            ]
            self.soft_update_xi_bar = tf.group(*soft_update_xi_bar_ops)
            # self.check = tf.add_check_numerics_ops()
            self.entropy = tf.reduce_mean(self.entropy_from_params(self.parameters))
            # ensure that xi and xi_bar are the same at initialization

            # TD error prediction model
            key_dim = (list(o_shape)[0] +  # o1
                       list(a_shape)[0]  # a
                       )
            # 1 +                 # r
            # list(o_shape)[0] +  # o2
            # 1)                  # t

            self.history = tf.placeholder(tf.float32, [batch_size, key_dim])
            oa = tf.concat([self.O1, self.A], axis=1)
            self.old_delta_tde = tf.placeholder(tf.float32, [batch_size])
            self.delta_tde = tf.placeholder(tf.float32, [batch_size])

            with tf.variable_scope('tde_model'):

                _batch_size = 32
                history = tf.reshape(self.history, [1, key_dim * _batch_size])
                old_delta_tde = tf.reshape(self.old_delta_tde, [1, _batch_size])
                past_mapping = tf.concat([history, old_delta_tde, ], axis=1)
                past_mapping = tf.tile(past_mapping, [_batch_size, 1])
                inputs = tf.concat([
                    oa, past_mapping
                ], axis=1)
                estimated_delta = tf.layers.dense(self.network(inputs).output, 1)
                self.estimated_delta = tf.reduce_mean(estimated_delta)

            self.model_loss = tf.reduce_mean(tf.square(estimated_delta - self.delta_tde))
            self.train_model, self.model_grad = train_op(
                loss=self.model_loss,
                var_list=[v for scope in ['tde_keys', 'tde_values', 'tde_model']
                          for v in get_variables(scope)])

            sess.run(tf.global_variables_initializer())

            # ensure that xi and xi_bar are the same at initialization
            hard_update_xi_bar_ops = [tf.assign(xbar, x) for (xbar, x) in zip(xi_bar, xi)]

            hard_update_xi_bar = tf.group(*hard_update_xi_bar_ops)
            sess.run(hard_update_xi_bar)

    @property
    def seq_len(self):
        return self._seq_len

    def train_step(self, step: Step) -> dict:
        feed_dict = {
            self.O1: step.o1,
            self.A:  step.a,
            self.R:  np.array(step.r) * self.reward_scale,
            self.O2: step.o2,
            self.T:  step.t,
        }
        return self.sess.run({attr: getattr(self, attr)
                              for attr in self.default_train_values}, feed_dict)

    def get_actions(self, o: ArrayLike, sample: bool = True, state=None) -> NetworkOutput:
        A = self.A_sampled1 if sample else self.A_max_likelihood
        return NetworkOutput(output=self.sess.run(A, {self.O1: [o]})[0], state=0)

    def pi_network(self, o: tf.Tensor) -> NetworkOutput:
        with tf.variable_scope('pi'):
            return self.network(o)

    def q_network(self, o: tf.Tensor, a: tf.Tensor, name: str,
                  reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            oa = tf.concat([o, a], axis=1)
            return tf.reshape(tf.layers.dense(self.network(oa).output, 1, name='q'), [-1])

    def v_network(self, o: tf.Tensor, name: str, reuse: bool = None) -> tf.Tensor:
        with tf.variable_scope(name, reuse=reuse):
            return tf.reshape(tf.layers.dense(self.network(o).output, 1, name='v'), [-1])

    def get_v1(self, o1: np.ndarray):
        return self.sess.run(self.v1, feed_dict={self.O1: [o1]})[0]

    def td_error(self, step: Step):
        return self.sess.run(
            self._td_error,
            feed_dict={
                self.O1: step.o1,
                self.A:  step.a,
                self.R:  step.r,
                self.O2: step.o2,
                self.T:  step.t
            })

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
