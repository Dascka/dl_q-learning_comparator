import gym
import numpy as np
import random
import os
from collections import namedtuple
import time

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class RandomAgent:
    '''Random agent'''
    def __init__(self, config):
        self.config = config
        self.env = gym.make("CartPole-v0")
        np.random.seed(config.seed)
        self.env.seed(config.seed)
        random.seed(config.seed)
        os.environ['PYTHONHASHSEED']=str(config.seed)

    def select_action(self):
        return np.random.choice(self.env.action_space.n, 1)

    def play_n(self, n_steps):

        returns = []
        steps = []
        sum_rewards = 0
        episode = []

        state = self.env.reset()

        for step in range(int(n_steps)):
            action = self.select_action()
            next_state, reward, done, _ = self.env.step(int(action)) #action.cpu() in source

            if not done:
                state = next_state
                sum_rewards += reward
                episode.append(Transition(state, action, reward, next_state, done))
            else:
                state = self.env.reset()
                steps.append(step)
                returns.append(sum_rewards)
                sum_rewards = 0
                # self.config.graph_special(episode, step)
                episode = []

        return steps, returns

def train(config):

    #to get total time of training
    start_time = time.time()  

    agent = RandomAgent(config)
    episode = config.episode
    max_steps_ep = config.max_steps
    max_steps_total = episode * max_steps_ep
    steps, returns = agent.play_n(max_steps_total)

    total_time = time.time() - start_time

    return steps, returns, total_time