import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from data_preprocessing.data_preprocessing import MountainCarNew


class DCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units])]
        self.output_size = [tf.TensorShape([units]), tf.TensorShape([units])]
        super(DCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        # expect input_shape (batch * 3 (pos, vel, action))

        self.kernel_state = self.add_weight(
            shape=(2, self.units), initializer="uniform", name="kernel_state"
        )
        self.kernel_hidden = self.add_weight(
            shape=(self.units, self.units), initializer="uniform", name="kernel_hidden"
        )
        self.kernel_action = self.add_weight(
            shape=(1, self.units),
            initializer="uniform",
            name="kernel_action",
        )
        self.bias = self.add_weight(
            shape=(1, self.units),
            initializer="zeros",
            name="bias_action",
        )

    def call(self, inputs, states):
        # inputs should be (batch, 3(pos, vel, action))

        input_1, input_2 = inputs[:, :2], inputs[:, 2:]
        s1, s2 = states

        s1 = tf.matmul(input_1, self.kernel_state) + tf.matmul(s2, self.kernel_hidden)
        s2 = tf.matmul(input_2, self.kernel_action) + self.bias + s1
        s2 = keras.activations.tanh(s2)
        return (s1, s2), (s1, s2)


class DModel(keras.Model):

    def __init__(self, mean_in, std_in, mean_out, std_out, hidden_size=40, target_size=1, dropout=0.001, **kwargs):
        super(DModel, self).__init__(**kwargs)
        self.env = MountainCarNew()
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.dropout_rate = dropout
        self.norm_in = layers.Lambda(lambda x: ((x - mean_in) / std_in), dtype=np.float64)
        self.rnn = layers.RNN(DCell(hidden_size))
        self.dropout = layers.Dropout(dropout)
        self.dense = layers.Dense(target_size * 2, activation='linear')
        self.norm_out = layers.Lambda(lambda x: ((x * std_out) + mean_out))

    def call(self, inputs):
        inputs = self.norm_in(inputs)
        output1, output2 = self.rnn(inputs)
        output2 = self.dropout(output2)
        next_state = self.dense(output2)
        next_state = self.norm_out(next_state)

        pos, vel = next_state[:, 0], next_state[:, 1]
        pos = tf.clip_by_value(pos, self.env.min_position, self.env.max_position)
        vel = tf.clip_by_value(vel, -self.env.max_speed, self.env.max_speed)
        next_state = tf.stack([pos, vel], axis=1)

        return next_state

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout': self.dropout_rate,
            'target_size': self.target_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


