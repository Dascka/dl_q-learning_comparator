"""Source : adaptation from https://github.com/flyyufelix/VizDoom-Keras-RL"""
#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, add, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from agents.Vizdoom.networks import Networks


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img
    

class DoubleDQNAgent:

    def __init__(self, state_size, action_size, config):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = config.discount
        self.learning_rate = config.lr_init
        self.epsilon = config.epsilon
        self.initial_epsilon = config.epsilon
        self.final_epsilon = config.epsilon_min
        self.batch_size = config.batch_size
        self.observe = config.warmup
        self.epsilon_decay = config.epsilon_decay
        self.frame_per_action = 4
        self.update_target_freq = config.update_target #3000
        self.timestep_per_train = 1 # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = deque()
        self.max_memory = config.window_size # number of previous transitions to remember

        # create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]): # Use ammo
            r_t = r_t - 0.1

        return r_t

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        # if self.epsilon > self.final_epsilon and t > self.observe:
        #     self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros(((batch_size,) + self.state_size)) # Shape 64, img_rows, img_cols, 4
        update_target = np.zeros(((batch_size,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input) # Shape 64, Num_Actions

        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1]), loss

    # Pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        
        minibatch = random.sample(self.memory, num_samples)
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

        loss = self.model.fit(states, targets_full, epochs=1, verbose=0)

        return np.max(targets[-1]), loss.history['loss']

def train(conf):

    #to get total time of training
    start_time = time.time()    

    #set the seeds for reproductability
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    tf.set_random_seed(conf.seed)

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("VizDoom/scenarios/defend_the_center.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.set_living_reward(0.1)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows , img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4 # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size, conf)

    agent.model = Networks.dqn(state_size, action_size, agent.learning_rate)
    agent.target_model = Networks.dqn(state_size, action_size, agent.learning_rate)

    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    is_terminated = game.is_episode_finished()

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0 # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics 
    life_buffer, ammo_buffer, kills_buffer = [], [], [] 

    scores, episodes, steps, kills, ammos = [], [], [], [], []
    step = 0
    episode = conf.episode
    e = 0
    score = 0
    while e < episode:
        loss = 0
        Q_max = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t)
        a_t[action_idx] = 1
        a_t = a_t.astype(int)

        r_t = game.make_action(a_t.tolist(), agent.frame_per_action)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        # print(r_t)
        score += r_t
        step += 1

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            ammo_buffer.append(misc[1])
            kills_buffer.append(misc[0])
            kills.append(misc[0])
            ammos.append(misc[1])
            print ("Episode Finish ", misc)
            # print(scores)
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

            scores.append(score)
            score = 0
            steps.append(step)
            episodes.append(e)
            e += 1
            

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        # Update the cache
        prev_misc = misc

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()
            
        s_t = s_t1
        t += 1

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and agent.epsilon > agent.final_epsilon:
            state = "explore"
        else:
            state = "train"

        if (is_terminated):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", score, \
                  "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe: 
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer, ammo_buffer, kills_buffer = [], [], [] 

    total_time = time.time() - start_time

    return steps, scores, total_time, kills, ammos