import numpy as np
import pandas as pd

from collections import deque
from random import sample

import tensorflow as tf
from tensorflow import keras
# from keras.optimizers import Adam

import config

class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_model(self, trainable=True):
        num_inputs = 10
        num_actions = 3
        num_hidden = 256

        '''
        model = tf.keras.models.Sequential()

        model.add(keras.layers.Dense(units=256, activation='relu', input_dim=self.state_dim))
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dropout(.1))
        model.add(keras.layers.Dense(units=3, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        model.summary()
        print('output shape:', model.output_shape)

        return model
        '''

        # Marche pas vraiment: Mettre trainable=True car "the model not being trainable, because if the weights of the model cannot be updated the optimization loop fails"
        # I was able to fix this, the reason why it was throwing an error was my batch_size was set greater than the size of the whole dataset.
        inputs = keras.layers.Input(shape=(num_inputs,))
        layer1 = keras.layers.Dense(num_hidden, activation="relu", trainable=trainable)(inputs)
        layer2 = keras.layers.Dense(num_hidden, activation="relu", trainable=trainable)(layer1)
        layer3 = keras.layers.Dropout(0.1, trainable=trainable)(layer2)
        action = keras.layers.Dense(num_actions, activation="softmax", trainable=trainable)(layer3)

        critic = keras.layers.Dense(1, trainable=trainable)(action)

        if config.CRITIC_ON:
            model = keras.Model(inputs=inputs, outputs=[action, critic])
        else:
            model = keras.Model(inputs=inputs, outputs=action)

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=opt)
        # model.compile(loss='categorical_crossentropy', optimizer=opt)
        # model.compile(loss='mse', optimizer=Adam(lr=0.001))

        model.summary()
        print('output shape:', model.output_shape)

        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)

        state_shape = state.shape

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        # q = self.online_network.predict(state)
        # return np.argmax(q, axis=1).squeeze()
        # return np.argmax(q[0], axis=1).squeeze()
        if config.CRITIC_ON:
            action_probs, critic_value = self.online_network.predict(state)
        else:
            action_probs = self.online_network.predict(state)
            critic_value = 0

        # return np.argmax(action_probs, axis=1).squeeze()
        return np.argmax(action_probs, axis=1)[0]

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return

        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        i = self.episode_length
        j = self.episodes

        # next_states[next_states == 0] = 0.01

        next_states = tf.convert_to_tensor(next_states)
        # next_states = tf.expand_dims(next_states, 0)

        next_q_values = self.online_network.predict_on_batch(next_states)

        next_q_values = tf.convert_to_tensor(next_q_values)

        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)

        next_q_values_target = tf.convert_to_tensor(next_q_values_target)

        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        states = tf.convert_to_tensor(states)

        q_values = self.online_network.predict_on_batch(states)

        q_values = q_values

        idx = np.array(self.idx)
        targets = np.array(targets)

        # CEDE Comment - Code to be fixed:
        for i in idx:
            q_values[i][actions[i]] = targets[i]

        # q_values[[self.idx, actions]] = targets
        # q_values[[idx, actions]] = targets

        q_values = tf.convert_to_tensor(q_values)

        loss = self.online_network.train_on_batch(x=states, y=q_values)

        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()


