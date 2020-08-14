import gym
import numpy as np
import random
import os
from collections import namedtuple
import time
from vizdoom import DoomGame, ScreenResolution
from vizdoom import *

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class RandomAgent:
    '''Random agent'''
    def __init__(self, action_size, config):
        self.config = config
        self.env = gym.make("LunarLander-v2")
        np.random.seed(config.seed)
        self.env.seed(config.seed)
        random.seed(config.seed)
        os.environ['PYTHONHASHSEED']=str(config.seed)
        self.action_size = action_size

    def select_action(self):
        return random.randrange(self.action_size)

    def play_n(self, n_steps):
        pass

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]): # Use ammo
            r_t = r_t - 0.1

        # if (misc[2] < prev_misc[2]): # Loss HEALTH
        #     r_t = r_t - 0.1

        return r_t

def train(conf):
    #to get total time of training
    start_time = time.time()  

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

    agent = RandomAgent(action_size, conf)

    episode = conf.episode


    # Start training
    GAME = 0
    t = 0
    max_life = 0 # Maximum episode life (Proxy for agent performance)
    life = 0

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
        action_idx = agent.select_action()

        a_t[action_idx] = 1
        a_t = a_t.astype(int)

        r_t = game.make_action(a_t.tolist(), 4)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        score += r_t
        step += 1

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
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

        misc = game_state.game_variables
        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        # Update the cache
        prev_misc = misc

        t += 1

    total_time = time.time() - start_time

    return steps, scores, total_time, kills, ammos

    # return steps, returns, total_time