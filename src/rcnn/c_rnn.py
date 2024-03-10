from tensorflow import keras
import tensorflow as tf
import numpy as np
from data_preprocessing.data_preprocessing import MountainCarNew


class CModel(keras.Model):
    def __init__(self, d_model, window_size, mean_in, std_in, mean_out, std_out, mean_a, std_a):
        super(CModel, self).__init__()

        self.env = MountainCarNew()
        self.window_size = window_size

        # copy from d_nn
        self.d_rnn = d_model.layers[1]
        self.d_rnn.trainable = False
        self.dropout = d_model.layers[2]
        self.dropout.trainable = False
        self.d_dense = d_model.layers[3]
        self.d_dense.trainable = False

        self.dense = keras.layers.Dense(units=1, activation='tanh')
        self.norm_in = keras.layers.Lambda(lambda x: ((x - mean_in) / std_in), dtype=np.float64)
        self.norm_out = keras.layers.Lambda(lambda x: ((x * std_out) + mean_out))
        self.norm_a = keras.layers.Lambda(lambda x: ((x * std_a) + mean_a))

    def call(self, inputs):

        # inputs of shape (batch_size * window_size * (pos, vel, action))

        inputs = self.norm_in(inputs)
        s1, s2 = self.d_rnn(inputs)
        a = self.dense(s1)
        a = self.norm_a(a)

        # calculate next state
        d_rnn_weights = self.d_rnn.weights
        new_s2 = tf.matmul(a, d_rnn_weights[2]) + d_rnn_weights[3] + s1
        next_state = self.dropout(new_s2)
        next_state = self.d_dense(next_state)
        next_state = self.norm_out(next_state)
        pos, vel = next_state[:, 0], next_state[:, 1]

        pos = tf.clip_by_value(pos, self.env.min_position, self.env.max_position)
        vel = tf.clip_by_value(vel, -self.env.max_speed, self.env.max_speed)
        next_state = tf.stack([pos, vel], axis=1)

        # get next action
        next_s1 = tf.matmul(next_state, d_rnn_weights[0]) + tf.matmul(s2, d_rnn_weights[1])
        next_action = self.dense(next_s1)
        next_action = self.norm_a(next_action)

        # reward
        rewards = 1 / (1 + tf.math.exp(-10 * (next_state[:, 0] - 0.5)))
        loss = - tf.reduce_mean(rewards)
        self.add_loss(loss)

        return a, next_state, next_action, rewards

