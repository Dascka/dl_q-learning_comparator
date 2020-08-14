""" Source : https://github.com/nilportugues/reinforcement-learning-1"""

import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow import set_random_seed
import os

class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space, config):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = config.epsilon
        self.gamma = config.discount
        self.batch_size = config.batch_size
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.lr_init
        self.memory = deque(maxlen=config.window_size)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train(config):
    
    #to get total time of training
    start_time = time.time()  

    env = gym.make('CartPole-v0')
    env.seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    set_random_seed(config.seed)
    os.environ['PYTHONHASHSEED']=str(config.seed)

    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0], config)
    episode = config.episode
    step = 0
    steps = []
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 4))
        score = 0
        max_steps = config.max_steps
        for i in range(max_steps):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 4))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            step += 1
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        steps.append(step)

    total_time = time.time() - start_time

    return steps, loss, total_time
