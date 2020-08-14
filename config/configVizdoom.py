import numpy as np
# from Benchmarks.DQN.models import Network
import matplotlib.pyplot as plt
from pathlib import Path

class Config:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ###Parameters
        self.epsilon = 1.0 #Initial value of epsilon
        self.epsilon_decay = 0.9999 #Number of training steps before epsilon reach 0. If 0 epsilon stay constant.
        self.epsilon_min = 0.0001
        self.discount = 0.99  # Chronological discount of the reward
        self.warmup = 5000
        self.training_steps = 120000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.window_size = 50000  # Number of self-play games to keep in the replay buffer
        self.update_target = 3000 #Number of training update before the target network will be updated 
        self.episode = 1500

        # Exponential learning rate schedule
        self.lr_init = 0.0001  # Initial learning rate

        ### Game
        self.observation_shape = (1, 1, 4)  # Dimensions of the game observation, must be 3D. For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(2)]  # Fixed list of all possible actions. You should only edit the length

        # self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available
        self.training_device = "cpu"