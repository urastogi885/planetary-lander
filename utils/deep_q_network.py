import random
import pickle
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.activations import relu, linear


class DeepQNetwork:
    # A class to implement the deep q-learning algorithm

    def __init__(self, state_space, action_space, alpha=0.001, gamma=0.99, epsilon=1.0):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 50
        self.memory = deque(maxlen=10000000)
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        return model

    def get_optimal_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_memory(self):
        memory_file = open('memory_discrete.npy', 'wb')
        pickle.dump(self.memory, memory_file)

    def load_memory(self):
        memory_file = open('memory_discrete.npy', 'rb')
        self.memory = pickle.load(memory_file)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in mini_batch])
        actions = np.array([i[1] for i in mini_batch])
        rewards = np.array([i[2] for i in mini_batch])
        next_states = np.array([i[3] for i in mini_batch])
        dones = np.array([i[4] for i in mini_batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
