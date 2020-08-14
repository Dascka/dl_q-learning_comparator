import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras import backend as k
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Lambda, add, Input
import time

#Source : https://github.com/nilportugues/reinforcement-learning-1


# this is Dueling DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DuelingDQNAgent:
    def __init__(self, state_size, action_size, config):
        # if you want to see Cartpole learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = config.discount
        self.learning_rate = config.lr_init
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.batch_size = config.batch_size
        self.train_start = config.warmup
        # create replay memory using deque
        self.memory = deque(maxlen=config.window_size)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    # the key point of Dueling network
    # the network devided into two streams, 1. value function 2. advantaget function
    # at the end of network, two streams are merged into one output stream which is Q function
    def build_model(self):
        input = Input(shape=(self.state_size,))
        x = Dense(150, input_shape=(self.state_size,), activation='relu', kernel_initializer='he_uniform')(input)
        x = Dense(120, activation='relu', kernel_initializer='he_uniform')(x)

        state_value = Dense(1, kernel_initializer='he_uniform')(x)
        state_value = Lambda(lambda s: k.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)

        action_advantage = Dense(self.action_size, kernel_initializer='he_uniform')(x)
        action_advantage = Lambda(lambda a: a[:, :] - k.mean(a[:, :], keepdims=True),
                                  output_shape=(self.action_size,))(action_advantage)

        q_value = add([state_value, action_advantage])
        model = Model(input=input, output=q_value)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

        # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

def train(config):
    #to get total time of training
    start_time = time.time()

    env = gym.make('LunarLander-v2')
    env.seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DuelingDQNAgent(state_size, action_size, config)

    scores, episodes, steps = [], [], []
    step = 0
    episode = config.episode
    for e in range(episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # agent.load_model("./save_model/cartpole-master.h5")

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            # reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_replay()
            score += reward
            state = next_state
            step += 1
            if done:
                env.reset()
                # every episode update the target model to be same with model

                agent.update_target_model()
                # every episode, plot the play time
                score = score if score == 499 else score + 100
                scores.append(score)
                steps.append(step)
                episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/Cartpole_Dueling_DQN.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # CURRENTLY DO NOT stop training
                if len(scores) > 100 and np.mean(scores[-100:]) > 200:
                    total_time = time.time() - start_time
                    return steps, scores, total_time

    total_time = time.time() - start_time

    return steps, scores, total_time