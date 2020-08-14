import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Config:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ###Parameters
        self.epsilon = 1 #Initial value of epsilon
        self.epsilon_decay = 0.9999 #Number of training steps before epsilon reach 0. If 0 epsilon stay constant.
        self.epsilon_min = 0.01
        self.discount = 0.997  # Chronological discount of the reward
        self.warmup = 20000
        self.training_steps = 120000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.window_size = 100000  # Number of self-play games to keep in the replay buffer
        self.update_target = 400 #Number of training update before the target network will be updated 
        self.max_steps = 1000
        self.episode = 1000

        # Exponential learning rate schedule
        self.lr_init = 0.00011111  # Initial learning rate

        self.training_device = "cpu"

        #Special values
        self.render_episode = 50 #Interval between each rendered episode. Set to 0 to not render any episode.
        self.count_episode = 0 #Do not change it counts the number of episodes encountered to do the renders.
        # self.path_to_graphs = "./Results/conf3/CartPole/VRAI_random/"

    def game_solved(self, return_list):
        return (len(return_list) >= 100 and np.mean(return_list[-100:]) > 195)

    def game_solved_ever(self, return_list):
        if len(return_list) < 100:
            return False
        
        solved = False
        for i in range(len(return_list) - 100):
            solved = solved or self.game_solved(return_list[i:i+100])
            if solved:
                return True
        
        return False

    # def graph_special(self, episode, training_step):
    #     '''
    #     For this game we plot the position of the cart for each time step.
    #     '''
    #     if (self.count_episode % self.render_episode) == 0: #we need to render the episode
    #         x_ax = []
    #         y_ax = []
    #         for i, transition in enumerate(episode):
    #             cart_pos = transition.state[0]
    #             x_ax.append(i)
    #             y_ax.append(cart_pos)

    #         fig, ax = plt.subplots(figsize = (20,20))
    #         ax.plot(x_ax, y_ax, color = 'b')

    #         ax.set_title("Episode number %g rendered at the %g training step." % (self.count_episode, training_step))
    #         ax.set_xlabel("Training steps")
    #         ax.set_ylabel("Cart position")

    #         Path(self.path_to_graphs).mkdir(parents=True, exist_ok=True)

    #         # plt.show()
    #         plt.savefig(self.path_to_graphs+str(self.count_episode)+".eps", format="eps")
    #         plt.close()

        
    #     self.count_episode += 1