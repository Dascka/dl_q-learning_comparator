from agents.OpenAI.LunarLander.dqn import train as DQN
from agents.OpenAI.LunarLander.ddqn import train as DDQN
from agents.OpenAI.LunarLander.dueling import train as Dueling_DQN
from agents.OpenAI.LunarLander.dueling_ddqn import train as Dueling_DDQN
from agents.OpenAI.LunarLander.random import train as random
# from Benchmarks.random.random import RandomAgent
from utils.graphics import step_reward_plot, ep_return_mean_plot, ep_avgx_mean_plot, ep_avgx_mean_plot_multiple
from utils.csv_reading import write_step_return_csv as write_csv
from utils.csv_reading import read_from_one_csv as read_csv
from pathlib import Path
import os
import config.configOpenAI as config
import sys, getopt

def get_hyperparam(algo : int = 0):
    '''
    Return all the hyperparameters tested by the algorithm corresponding to the param algo
    :param algo: Index of the algorithm. 0 is DQN, 1 is DDQN, 2 is Dueling DQN, 3 is Dueling DDQN, -1 is random.
    '''

    if algo == 0:
        lrs = [0.001, 0.00025]
        discounts = [0.99, 0.95]
        epsilons = [1]
        eps_decays = [0.996, 0.9999]
        eps_mins = [0.1]
        batch_sizes = [64]
        max_steps = [3000]
        warmups = [1000]
    elif algo == 1:
        lrs = [0.001, 0.00025]
        discounts = [0.99, 0.95]
        epsilons = [1]
        eps_decays = [0.996, 0.9999]
        eps_mins = [0.1]
        batch_sizes = [64]
        warmups = [1000]
        max_steps = [1000]
    elif algo == 2:
        lrs = [0.001, 0.00025]
        discounts = [0.99, 0.95]
        epsilons = [1]
        eps_decays = [0.996, 0.9999]
        eps_mins = [0.1]
        batch_sizes = [64]
        warmups = [1000]
        max_steps = [3000]
    elif algo == 3:
        lrs = [0.00025, 0.001]
        discounts = [0.999, 0.95]
        epsilons = [1]
        eps_decays = [0.9, 0.9999]
        eps_mins = [0.01]
        batch_sizes = [64]
        max_steps = [1000]
        warmups = [1000]
    elif algo == -1:
        lrs = [0]
        discounts = [0]
        epsilons = [0]
        eps_decays = [0]
        eps_mins = [0]
        batch_sizes = [0]
        max_steps = [1000]
        warmups = [0]

    return lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups

def train(start: int, stop: int, repeat: int, start_repeat : int = 0, algo = 0):
    '''
    Function for benchmarking for OpenAI LunarLander. We can precise which starting point we want wrt the hyperparameters selectionned and the number of repeat we want. Algo is a number that precise which algorithm we are benchmarking.
    :param start: Starting point in the list of hyperparameters.
    :param end: Ending point in the list of hyperparameters.
    :param repeat: Ending point of repeat.
    :param start_repeat: Starting point of repeat, we are proceding this way to fix the seeds correctly.
    :param algo: 0 is DQN. 1 is DDQN. 2 is Dueling DQN. 3 is Dueling DDQN. Otherwise it is DQN.
    '''

    agent = DQN

    if algo == 0:
        agent = DQN
        algname = "DQN"
    elif algo == 1:
        agent = DDQN
        algname = "DDQN"
    elif algo == 2:
        agent = Dueling_DQN
        algname = "Dueling_DQN"
    elif algo == 3:
        agent = Dueling_DDQN
        algname = "Dueling_DDQN"
    elif algo == -1:
        agent = random
        algname = "Aléatoire"
        lrs = [0]

    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo)

    conf = config.Config()

    i = 0

    for lr in lrs:
        for discount in discounts:
            for epsilon in epsilons:
                for eps_decay in eps_decays:
                    for eps_min in eps_mins:
                        for batch_size in batch_sizes:
                            for max_step in max_steps:
                                for warmup in warmups:
                                    conf.lr_init = lr
                                    conf.discount = discount
                                    conf.epsilon = epsilon
                                    conf.epsilon_decay = eps_decay
                                    conf.epsilon_min = eps_min
                                    conf.batch_size = batch_size
                                    conf.max_steps = max_step
                                    conf.warmup = warmup
                                    if i >= start and i < stop:
                                        for l in range(start_repeat, repeat):
                                            config.seed = l
                                            path = "./Results/LunarLander/" + algname + "/graph/"+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                            path_csv = "./Results/LunarLander/" + algname + "/graph/"+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"
                                            Path("./Results/LunarLander/" + algname + "/graph/"+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/").mkdir(parents=True, exist_ok=True)
                                            Path("./Results/LunarLander/" + algname + "/graph/"+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/").mkdir(parents=True, exist_ok=True)


                                            steps, returns, time = agent(conf)

                                            title = "%s :\n eps %g, eps_dec %g, eps_min %g,\n gamma %g, ep %g, batch_s %g,\n mem_s %g,\n lr_i %g, \n time %g" %(algname, conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init, time)
                                            step_reward_plot([steps], [returns], title, [algname], path)

                                            write_csv(steps, returns, path_csv)
                                    i += 1

def graphs(start_path = "Results/LunarLander/", algo = 0, repeat = 20, start_repeat = 0):
    '''
    Function for benchmarking for OpenAI LunarLander that creates the graphs.
    :param path: Start of the path where we can find the csv and will save eps files
    :param algo: Number defining for which algo we are drawing the graphs.
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''

    ylim = [-500, 500]
    win_cond = 200
    stop_cond = True
    
    if algo == 0:
        algname = "DQN"
    elif algo == 1:
        algname = "DDQN"
    elif algo == 2:
        algname = "Dueling_DQN"
    elif algo == 3:
        algname = "Dueling_DDQN"
    elif algo == -1:
        algname = "Aléatoire"
    
    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo)

    conf = config.Config()
    start_path += algname + "/graph/"

    i = 0

    for lr in lrs:
        for discount in discounts:
            for epsilon in epsilons:
                for eps_decay in eps_decays:
                    for eps_min in eps_mins:
                        for batch_size in batch_sizes:
                            for max_step in max_steps:
                                for warmup in warmups:
                                    conf.lr_init = lr
                                    conf.discount = discount
                                    conf.epsilon = epsilon
                                    conf.epsilon_decay = eps_decay
                                    conf.epsilon_min = eps_min
                                    conf.batch_size = batch_size
                                    conf.max_steps = max_step
                                    conf.warmup = warmup

                                    steps = []
                                    returns = []
                                    for l in range(start_repeat, repeat):
                                        config.seed = l
                                        path = start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                        path_csv = start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"

                                        tmp_step, tmp_returns = read_csv(path_csv)
                                        steps.append(tmp_step)
                                        returns.append(tmp_returns)

                                    title = algname + " :\n eps %g, eps_dec %g, eps_min %g,\n gamma %g, ep %g, batch_s %g,\n mem_s %g,\n lr_i %g, \n game_solv %r" %(conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init, conf.game_solved_ever(returns))
                                    if algo == -1: 
                                        title = "Résultats de l'algorithme aléatoire pour LunarLander"

                                    path = start_path +"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/mean_not_labeled_"+ str(i) +".pdf"
                                    ep_return_mean_plot(returns, title, path, False, ylim, True, stop_cond, win_cond, xlim = [0, 400])
                                    path =start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/mean_labeled_" + str(i) +".pdf"
                                    ep_return_mean_plot(returns, title, path, True, ylim, True, stop_cond, win_cond, xlim = [0, 400])

                                    if algo == -1: 
                                        title = "Résultats de l'algorithme aléatoire pour VizDoom \n Moyenne sur les cent derniers épisodes"

                                    path =start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_mean_not_labeled_"+ str(i) +".pdf"
                                    ep_avgx_mean_plot(returns, False, title, path, False, ylim, True, repeat, stop_cond, win_cond, [0, 400])
                                    path =start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_mean_labeled_" + str(i) +".pdf"
                                    ep_avgx_mean_plot(returns, False, title, path, True, ylim, True, repeat, stop_cond, win_cond, [0, 400])

                                    i += 1

def multi_graphs(start_path = "Results/LunarLander/", DQN_n = 0, DDQN_n = 0, Dueling_n = 0, Dueling_DDQN_n = 0, repeat = 20, start_repeat = 0):
    '''
    Function for benchmarking for OpenAI LunarLander that creates the graphs.
    :param path: Start of the path where we can find the csv and will save eps files. Should be the path before the algorithms folders supposed that the algorithms' folders are named "DQN", "DDQN", "Dueling_DQN", "Dueling_DDQN" and "Aléatoire" followed by a folder named graph.
    :param ALGO_n: Define the selection of hyperparamers that we will consider. ALGO is to be replaced with DQN, DDQN, Dueling and Dueling_DDQN
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''

    ylim = [-500, 500]
    xlim = [0, 400]
    win_cond = 200
    stop_cond = True

    #DQN part

    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo = 0)

    DQN_returns, DQN_steps = multi_graphs_reading_single(start_path=start_path+"DQN/graph/", index=DQN_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, max_steps=max_steps, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #DDQN
    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo = 1)

    DDQN_returns, DQN_steps = multi_graphs_reading_single(start_path=start_path+"DDQN/graph/", index=DDQN_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, max_steps=max_steps, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #Dueling DQN

    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo = 2)

    Dueling_returns, Dueling_steps = multi_graphs_reading_single(start_path=start_path+"Dueling_DQN/graph/", index=Dueling_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, max_steps=max_steps, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #Dueling DDQN

    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo = 3)

    Dueling_DDQN_returns, Dueling_DDQN_steps = multi_graphs_reading_single(start_path=start_path+"Dueling_DDQN/graph/", index=Dueling_DDQN_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, max_steps=max_steps, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #Aléatoire

    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo = -1)

    random_returns, random_steps = multi_graphs_reading_single(start_path=start_path+"Aléatoire/graph/", index=0, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, max_steps=max_steps, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    path = start_path + "Multi/graph/DQN" + str(DQN_n) + "-DDQN" + str(DDQN_n) + "-Dueling" + str(Dueling_n) + "-DuelingDDQN" + str(Dueling_DDQN_n) + ".pdf"
    
    Path(start_path + "Multi/graph/").mkdir(parents=True, exist_ok=True)

    ep_avgx_mean_plot_multiple([DQN_returns, DDQN_returns, Dueling_returns, Dueling_DDQN_returns, random_returns], title = "Comparaison d'algorithmes sur LunarLander", algnames = ["DQN", "DDQN", "Dueling DQN", "Dueling DDQN", "Aléatoire"], path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond, xlim = xlim)

def multi_graphs_reading_single(start_path, index = 0, lrs = [0.001, 0.00025], discounts = [0.999, 0.95], epsilons = [1], eps_decays = [0.9, 0.9999], eps_mins = [0.01], batch_sizes = [32], max_steps = [1000], warmups = [1000], repeat = 20, start_repeat = 0):
    '''
    Function for benchmarking for OpenAI LunarLander that reads the data for one set of hyperparameters.
    :param path: Start of the path where we can find the csv and will save eps files. Should be the path before the algorithms folders supposed that the algorithms' folders are named "DQN", "DDQN", "Dueling_DQN", "Dueling_DDQN" and "Aléatoire" followed by a folder named graph.
    :param index: Index of the selected set of hyperparameter.
    :param lrs: List of learning rates.
    :param discounts: List of discounts.
    :param epsilons: List of epsilons.
    :param eps_decays: List of epsilon decay rates.
    :param eps_min: List of epsilon's minimum value.
    :param batch_sizes: List of batch's sizes.
    :param max_steps: List of number of maximum steps.
    :param warmups: List of number of warmup steps.
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''
    steps = []
    returns = []

    i = 0

    conf = config.Config()

    for lr in lrs:
        for discount in discounts:
            for epsilon in epsilons:
                for eps_decay in eps_decays:
                    for eps_min in eps_mins:
                        for batch_size in batch_sizes:
                            for max_step in max_steps:
                                for warmup in warmups:
                                    if i == index:
                                        conf.lr_init = lr
                                        conf.discount = discount
                                        conf.epsilon = epsilon
                                        conf.epsilon_decay = eps_decay
                                        conf.epsilon_min = eps_min
                                        conf.batch_size = batch_size
                                        conf.max_steps = max_step
                                        conf.warmup = warmup

                                        for l in range(start_repeat, repeat):
                                            path = start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                            path_csv = start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"

                                            tmp_step, tmp_returns = read_csv(path_csv)
                                            steps.append(tmp_step)
                                            returns.append(tmp_returns)

                                    i += 1

    return returns, steps #return the returns and steps for every repeat of every set of hyperparameters specified

def multi_graphs_same_algo(start_path = "Results/LunarLander/", index = [0], algo = 0):
    '''
    Function for benchmarking for OpenAI LunarLander that creates the graphs.
    :param path: Start of the path where we can find the csv and will save eps files. Should be the path before the algorithms folders supposed that the algorithms' folders are named "DQN", "DDQN", "Dueling_DQN", "Dueling_DDQN" and "Aléatoire" followed by a folder named graph.
    :param index: Define the selection of hyperparamers that we will consider.
    :param alg: Index of the algorithme to work with the correct hyperparameters. 0 is DQN, 1 is DDQN, 2 is Dueling DQN, 3 is Dueling DDQN, -1 is random.
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''

    ylim = [-500, 500]
    xlim = [0, 400]
    win_cond = 200
    stop_cond = True
    
    if algo == 0:
        algname = "DQN"
    elif algo == 1:
        algname = "DDQN"
    elif algo == 2:
        algname = "Dueling_DQN"
    elif algo == 3:
        algname = "Dueling_DDQN"
    elif algo == -1:
        algname = "Aléatoire"

    lrs, discounts, epsilons, eps_decays, eps_mins, batch_sizes, max_steps, warmups = get_hyperparam(algo)

    returns, steps = multi_graphs_reading_multiple(start_path=start_path+ algname +"/graph/", index=index, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, max_steps=max_steps, warmups=warmups, repeat=repeat,start_repeat = start_repeat)

    path = start_path + "Multi/graph/" + algname + str(index).replace("[", "").replace(", ", "").replace("]", "") + ".pdf"
    
    Path(start_path + "Multi/graph/").mkdir(parents=True, exist_ok=True)

    algnames = [algname + str(ind) for ind in index]

    ep_avgx_mean_plot_multiple(returns, title = "Comparaison de " + algname + " sur LunarLander", algnames = algnames, path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond, xlim = xlim)

def multi_graphs_reading_multiple(start_path, index = [0], lrs = [0.001, 0.00025], discounts = [0.999, 0.95], epsilons = [1], eps_decays = [0.9, 0.9999], eps_mins = [0.01], batch_sizes = [32], max_steps = [1000], warmups = [1000], repeat = 20, start_repeat = 0):

    sel_returns = []
    sel_steps = []

    i = 0

    conf = config.Config()

    for lr in lrs:
        for discount in discounts:
            for epsilon in epsilons:
                for eps_decay in eps_decays:
                    for eps_min in eps_mins:
                        for batch_size in batch_sizes:
                            for max_step in max_steps:
                                for warmup in warmups:
                                    if i in index:
                                        conf.lr_init = lr
                                        conf.discount = discount
                                        conf.epsilon = epsilon
                                        conf.epsilon_decay = eps_decay
                                        conf.epsilon_min = eps_min
                                        conf.batch_size = batch_size
                                        conf.max_steps = max_step
                                        conf.warmup = warmup

                                        steps = []
                                        returns = []
                                        for l in range(start_repeat, repeat):
                                            path = start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                            path_csv = start_path+"lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"

                                            tmp_step, tmp_returns = read_csv(path_csv)
                                            steps.append(tmp_step)
                                            returns.append(tmp_returns)

                                        sel_steps.append(steps)
                                        sel_returns.append(returns)
                                    i += 1

    return sel_returns, sel_steps #return the returns and steps for every repeat of every set of hyperparameters specified

if __name__ == "__main__":
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, "s:e:r:", ["start=","end=","repeat=", "start_repeat="])
    start, end, repeat, start_repeat = 0, 0, 0, 0
    for opt, arg in opts:
        if opt in  ("-s", "--start"):
            start = arg
        elif opt in ('-e', "--end"):
            end = arg
        elif opt in ("-r", "--repeat"):
            repeat = arg
        elif opt == "--start_repeat":
            start_repeat = arg

    train(int(start), int(end), int(repeat), int(start_repeat))