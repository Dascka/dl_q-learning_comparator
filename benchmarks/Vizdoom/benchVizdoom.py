from agents.Vizdoom.ddqn import train as DDQN
from agents.Vizdoom.dqn import train as DQN
from agents.Vizdoom.dueling import train as Dueling_DQN
from agents.Vizdoom.dueling_ddqn import train as Dueling_DDQN
from agents.Vizdoom.random import train as random
from utils.graphics import step_reward_plot, ep_return_mean_plot, ep_avgx_mean_plot, ep_avgx_mean_plot_multiple
from utils.csv_reading import write_step_return_csv_vizdoom as write_csv
from utils.csv_reading import read_from_one_csv_vizdoom as read_csv
from pathlib import Path
import os
import config.configVizdoom as config
import sys, getopt

def f(start: int, stop: int, repeat: int, start_repeat : int = 0, algo = 0):
    '''
    Function for benchmarking for VizDoom. We can precise which starting point we want wrt the hyperparameters selectionned and the number of repeat we want. Algo is a number that precise which algorithm we are benchmarking.
    :param start: Starting point in the list of hyperparameters.
    :param end: Ending point in the list of hyperparameters.
    :param repeat: Ending point of repeat.
    :param start_repeat: Starting point of repeat, we are proceding this way to fix the seeds correctly.
    :param algo: 0 is DQN. 1 is DDQN. 2 is Dueling DQN. 3 is Dueling DDQN. Otherwise it is DQN.
    '''
    agent = DQN

    mems = [50000]
    lrs = [0.00025, 0.0001]
    discounts = [0.99]
    epsilons = [1]
    eps_decays = [0.99, 0.9]
    eps_mins = [0.0001]
    batch_sizes = [32]
    warmups = [50000]

    conf = config.Config()

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
        discounts = [0]
        epsilons = [0]
        eps_decays = [0]
        eps_mins = [0]
        batch_sizes = [0]
        warmups = [0]

    i = 0
    for mem in mems:
        for lr in lrs:
            for discount in discounts:
                for epsilon in epsilons:
                    for eps_decay in eps_decays:
                        for eps_min in eps_mins:
                            for batch_size in batch_sizes:
                                for warmup in warmups:
                                    conf.lr_init = lr
                                    conf.discount = discount
                                    conf.epsilon = epsilon
                                    conf.epsilon_decay = eps_decay
                                    conf.epsilon_min = eps_min
                                    conf.batch_size = batch_size
                                    conf.warmup = warmup
                                    if i >= start and i < stop:
                                        for l in range(start_repeat, repeat):
                                            config.seed = l
                                            path = "./Results/Vizdoom/" + algname + "/graph/mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                            path_csv = "./Results/Vizdoom/" + algname + "/graph/mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"
                                            Path("./Results/Vizdoom/" + algname + "/graph/mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/").mkdir(parents=True, exist_ok=True)
                                            Path("./Results/Vizdoom/" + algname + "/graph/mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/").mkdir(parents=True, exist_ok=True)

                                            print("eps %g, eps_dec %g, eps_min %g, gamma %g, ep %g, batch_s %g, mem_s %g, lr_i %g, game_solv" % (conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init))

                                            steps, returns, time, kills, ammos = agent(conf)

                                            title = "%s :\n eps %g, eps_dec %g, eps_min %g,\n gamma %g, ep %g, batch_s %g,\n mem_s %g,\n lr_i %g, \n time %g" %(algname, conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init, time)
                                            step_reward_plot([steps], [returns], title, [algname], path)

                                            write_csv(steps, returns, kills, ammos, path_csv)
                                    i += 1

def graphs(start_path = "Results/Vizdoom/", algo = 0, repeat = 10, start_repeat = 0):
    '''
    Function for benchmarking for VizDoom that creates the graphs.
    :param path: Start of the path where we can find the csv and will save eps files
    :param algo: Number defining for which algo we are drawing the graphs.
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''

    ylim = [0, 100]
    win_cond = 0
    stop_cond = False

    mems = [50000]
    lrs = [0.00025, 0.0001]
    discounts = [0.99]
    epsilons = [1]
    #0.9999977 il faut entraîner plus qu'1 millions d'étapes d'entraînement pour que ça soit worth
    eps_decays = [0.99, 0.9]
    eps_mins = [0.0001]
    batch_sizes = [32]
    warmups = [50000]
    
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
        mems = [0]
        lrs = [0]
        discounts = [0]
        epsilons = [0]
        eps_decays = [0]
        eps_mins = [0]
        batch_sizes = [0]
        warmups = [0]

    start_path += algname + "/graph/"
    conf = config.Config()

    i = 0

    for mem in mems:
        for lr in lrs:
            for discount in discounts:
                for epsilon in epsilons:
                    for eps_decay in eps_decays:
                        for eps_min in eps_mins:
                            for batch_size in batch_sizes:
                                for warmup in warmups:
                                    conf.lr_init = lr
                                    conf.discount = discount
                                    conf.epsilon = epsilon
                                    conf.epsilon_decay = eps_decay
                                    conf.epsilon_min = eps_min
                                    conf.batch_size = batch_size
                                    conf.warmup = warmup

                                    steps = []
                                    returns = []
                                    kills = []
                                    ammos = []
                                    for l in range(start_repeat, repeat):
                                        config.seed = l

                                        config.seed = l

                                        path = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                        path_csv = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"

                                        print("eps %g, eps_dec %g, eps_min %g, gamma %g, ep %g, batch_s %g, mem_s %g, lr_i %g, game_solv" % (conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init))

                                        tmp_step, tmp_returns, tmp_kills, tmp_ammos = read_csv(path_csv)
                                        steps.append(tmp_step)
                                        returns.append(tmp_returns)
                                        kills.append(tmp_kills)
                                        ammos.append(tmp_ammos)

                                    #Returns graphs
                                    ylim = [0, 100]

                                    title = algname + " :\n eps %g, eps_dec %g, eps_min %g,\n gamma %g, ep %g, batch_s %g,\n mem_s %g,\n lr_i %g" %(conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init)
                                    if algo == -1: 
                                        title = "Retours de l'algorithme aléatoire pour VizDoom"

                                    path = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/mean_not_labeled_"+ str(i) +".pdf"
                                    ep_return_mean_plot(returns, title, path, False, ylim, True, stop_cond, win_cond)
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/mean_labeled_" + str(i) +".pdf"
                                    ep_return_mean_plot(returns, title, path, True, ylim, True, stop_cond, win_cond)

                                    if algo == -1: 
                                        title = "Résultats de l'algorithme aléatoire pour VizDoom \n Moyenne sur les cent derniers épisodes"
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_mean_not_labeled_"+ str(i) +".pdf"
                                    ep_avgx_mean_plot(returns, False, title, path, False, ylim, True, repeat, stop_cond, win_cond)
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_mean_labeled_" + str(i) +".pdf"
                                    ep_avgx_mean_plot(returns, False, title, path, True, ylim, True, repeat, stop_cond, win_cond)

                                    #kills graphs

                                    ylim = [0, 30]

                                    title = algname + " Kills :\n eps %g, eps_dec %g, eps_min %g,\n gamma %g, ep %g, batch_s %g,\n mem_s %g,\n lr_i %g" %(conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init)
                                    if algo == -1: 
                                        title = "Ennemis tués de l'algorithme aléatoire pour VizDoom"

                                    path = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/kills_mean_not_labeled_"+ str(i) +".pdf"
                                    ep_return_mean_plot(kills, title, path, False, ylim, True, stop_cond, win_cond, ylabel = "Ennemis tués")
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/kills_mean_labeled_" + str(i) +".pdf"
                                    ep_return_mean_plot(kills, title, path, True, ylim, True, stop_cond, win_cond, ylabel = "Ennemis tués")

                                    if algo == -1: 
                                        title = "Ennemis tués de l'algorithme aléatoire pour VizDoom \n Moyenne sur les cent derniers épisodes"
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_kills_mean_not_labeled_"+ str(i) +".pdf"
                                    ep_avgx_mean_plot(kills, title, path, False, ylim, True, repeat, stop_cond, win_cond, ylabel = "Ennemis tués")
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_kills_mean_labeled_" + str(i) +".pdf"
                                    ep_avgx_mean_plot(kills, title, path, True, ylim, True, repeat, stop_cond, win_cond, ylabel = "Ennemis tués")

                                    #ammos graphs

                                    ylim = [0, 30]
                                    
                                    title = algname + " Ammos used :\n eps %g, eps_dec %g, eps_min %g,\n gamma %g, ep %g, batch_s %g,\n mem_s %g,\n lr_i %g" %(conf.epsilon, conf.epsilon_decay, conf.epsilon_min, conf.discount, conf.episode, conf.batch_size, conf.window_size, conf.lr_init)
                                    if algo == -1: 
                                        title = "Munitions utilisées de l'algorithme aléatoire pour VizDoom"

                                    path = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/ammos_mean_not_labeled_"+ str(i) +".pdf"
                                    ep_return_mean_plot(ammos, title, path, False, ylim, True, stop_cond, win_cond, ylabel = "Munitions utilisées")
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/ammos_mean_labeled_" + str(i) +".pdf"
                                    ep_return_mean_plot(ammos, title, path, True, ylim, True, stop_cond, win_cond, ylabel = "Munitions utilisées")

                                    if algo == -1: 
                                        title = "Munitions utilisées de l'algorithme aléatoire pour VizDoom \n Moyenne sur les cent derniers épisodes"
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_ammos_mean_not_labeled_"+ str(i) +".pdf"
                                    ep_avgx_mean_plot(ammos, title, path, False, ylim, True, repeat, stop_cond, win_cond, ylabel = "Munitions utilisées")
                                    path =start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/avg100_ammos_mean_labeled_" + str(i) +".pdf"
                                    ep_avgx_mean_plot(ammos, title, path, True, ylim, True, repeat, stop_cond, win_cond, ylabel = "Munitions utilisées")

                                    i += 1

def multi_graphs(start_path = "Results/Vizdoom/", DQN_n = 0, DDQN_n = 0, Dueling_n = 0, Dueling_DDQN_n = 0, repeat = 10, start_repeat = 0):
    '''
    Function for benchmarking for VizDoom that creates the graphs.
    :param path: Start of the path where we can find the csv and will save eps files. Should be the path before the algorithms folders supposed that the algorithms' folders are named "DQN", "DDQN", "Dueling_DQN", "Dueling_DDQN" and "Aléatoire" followed by a folder named graph.
    :param ALGO_n: Define the selection of hyperparamers that we will consider. ALGO is to be replaced with DQN, DDQN, Dueling and Dueling_DDQN
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''

    ylim = [0, 100]
    win_cond = 0
    stop_cond = False
    
    mems = [50000]
    lrs = [0.00025, 0.0001]
    discounts = [0.99]
    epsilons = [1]
    eps_decays = [0.99, 0.9]
    eps_mins = [0.0001]
    batch_sizes = [32]
    warmups = [50000]

    #DQN part

    DQN_returns, DQN_steps, DQN_kills, DQN_ammos = multi_graphs_reading_single(start_path=start_path+"DQN/graph/", index=DQN_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, mems=mems, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #DDQN

    DDQN_returns, DDQN_steps, DDQN_kills, DDQN_ammos = multi_graphs_reading_single(start_path=start_path+"DDQN/graph/", index=DDQN_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, mems=mems, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #Dueling DQN

    Dueling_returns, Dueling_steps, Dueling_kills, Dueling_ammos = multi_graphs_reading_single(start_path=start_path+"Dueling_DQN/graph/", index=Dueling_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, mems=mems, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #Dueling DDQN

    Dueling_DDQN_returns, Dueling_DDQN_steps, Dueling_DDQN_kills, Dueling_DDQN_ammos = multi_graphs_reading_single(start_path=start_path+"Dueling_DDQN/graph/", index=Dueling_DDQN_n, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, mems=mems, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #Aléatoire

    algname = "Aléatoire"
    mems = [0]
    lrs = [0]
    discounts = [0]
    epsilons = [0]
    eps_decays = [0]
    eps_mins = [0]
    batch_sizes = [0]
    warmups = [0]

    random_returns, random_steps, random_kills, random_ammos = multi_graphs_reading_single(start_path=start_path+"Aléatoire/graph/", index=0, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, mems=mems, warmups=warmups, repeat=repeat, start_repeat = start_repeat)

    #returns 

    path = start_path + "Multi/graph/returns_DQN" + str(DQN_n) + "-DDQN" + str(DDQN_n) + "-Dueling" + str(Dueling_n) + "-DuelingDDQN" + str(Dueling_DDQN_n) + ".pdf"
    
    Path(start_path + "Multi/graph/").mkdir(parents=True, exist_ok=True)

    ep_avgx_mean_plot_multiple([DQN_returns, DDQN_returns, Dueling_returns, Dueling_DDQN_returns, random_returns], title = "Comparaison d'algorithmes sur VizDoom\n Retours", algnames = ["DQN", "DDQN", "Dueling DQN", "Dueling DDQN", "Aléatoire"], path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond)

    #kills
    ylim = [0, 30]

    path = start_path + "Multi/graph/kills_DQN" + str(DQN_n) + "-DDQN" + str(DDQN_n) + "-Dueling" + str(Dueling_n) + "-DuelingDDQN" + str(Dueling_DDQN_n) + ".pdf"

    ep_avgx_mean_plot_multiple([DQN_kills, DDQN_kills, Dueling_kills, Dueling_DDQN_kills, random_kills], title = "Comparaison d'algorithmes sur VizDoom\n Ennemis tués", algnames = ["DQN", "DDQN", "Dueling DQN", "Dueling DDQN", "Aléatoire"], path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond)

    #ammos 
    ylim = [0, 30]

    path = start_path + "Multi/graph/ammos_DQN" + str(DQN_n) + "-DDQN" + str(DDQN_n) + "-Dueling" + str(Dueling_n) + "-DuelingDDQN" + str(Dueling_DDQN_n) + ".pdf"

    ep_avgx_mean_plot_multiple([DQN_ammos, DDQN_ammos, Dueling_ammos, Dueling_DDQN_ammos, random_ammos], title = "Comparaison d'algorithmes sur VizDoom\n Munititons restantes", algnames = ["DQN", "DDQN", "Dueling DQN", "Dueling DDQN", "Aléatoire"], path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond)

def multi_graphs_reading_single(start_path, index = 0, lrs = [0.001, 0.00025], discounts = [0.999, 0.95], epsilons = [1], eps_decays = [0.9, 0.9999], eps_mins = [0.01], batch_sizes = [32], mems = [1000], warmups = [1000], repeat = 10, start_repeat = 0):
    '''
    Function for benchmarking for VizDoom that reads the data for one set of hyperparameters.
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

    returns = []
    steps = []
    kills = []
    ammos = []

    i = 0

    conf = config.Config()

    for mem in mems:
        for lr in lrs:
            for discount in discounts:
                for epsilon in epsilons:
                    for eps_decay in eps_decays:
                        for eps_min in eps_mins:
                            for batch_size in batch_sizes:
                                for warmup in warmups:
                                    if i == index:
                                        conf.lr_init = lr
                                        conf.discount = discount
                                        conf.epsilon = epsilon
                                        conf.epsilon_decay = eps_decay
                                        conf.epsilon_min = eps_min
                                        conf.batch_size = batch_size
                                        conf.warmup = warmup

                                        for l in range(start_repeat, repeat):
                                            path = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                            path_csv = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"

                                            tmp_step, tmp_returns, tmp_kills, tmp_ammos = read_csv(path_csv)
                                            steps.append(tmp_step)
                                            returns.append(tmp_returns)
                                            kills.append(tmp_kills)
                                            ammos.append(tmp_ammos)

                                    i += 1

    return returns, steps, kills, ammos #return the returns and steps for every repeat of every set of hyperparameters specified

def multi_graphs_same_algo(start_path = "Results/Vizdoom/", index = [0], algo = 0):
    '''
    Function for benchmarking for VizDoom that creates the graphs.
    :param path: Start of the path where we can find the csv and will save eps files. Should be the path before the algorithms folders supposed that the algorithms' folders are named "DQN", "DDQN", "Dueling_DQN", "Dueling_DDQN" and "Aléatoire" followed by a folder named graph.
    :param index: Define the selection of hyperparamers that we will consider.
    :param alg: Index of the algorithme to work with the correct hyperparameters. 0 is DQN, 1 is DDQN, 2 is Dueling DQN, 3 is Dueling DDQN, -1 is random.
    :param start_repeat: Index where to start the repeat.
    :param repeat: Index where to end the repeat.
    '''

    ylim = [0, 100]
    xlim = [0, 400]
    win_cond = 0
    stop_cond = False

    mems = [50000]
    lrs = [0.00025, 0.0001]
    discounts = [0.99]
    epsilons = [1]
    #0.9999977 il faut entraîner plus qu'1 millions d'étapes d'entraînement pour que ça soit worth
    eps_decays = [0.99, 0.9]
    eps_mins = [0.0001]
    batch_sizes = [32]
    warmups = [50000]
    
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
        mems = [0]
        lrs = [0]
        discounts = [0]
        epsilons = [0]
        eps_decays = [0]
        eps_mins = [0]
        batch_sizes = [0]
        warmups = [0]

    returns, steps, kills, ammos = multi_graphs_reading_multiple(start_path=start_path+ algname +"/graph/", index=index, lrs = lrs, discounts=discounts, epsilons=epsilons, eps_decays=eps_decays, eps_mins = eps_mins, batch_sizes=batch_sizes, mems=mems, warmups=warmups, repeat=repeat,start_repeat = start_repeat)

    algnames = [algname + str(ind) for ind in index]

    path = start_path + "Multi/graph/returns_" + algname + str(index).replace("[", "").replace(", ", "").replace("]", "") + ".pdf"
    
    Path(start_path + "Multi/graph/").mkdir(parents=True, exist_ok=True)

    #Returns graph
    ep_avgx_mean_plot_multiple(returns, title = "Comparaison des retours pour " + algname + " sur VizDoom", algnames = algnames, path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond)

    #Kills graph
    path = start_path + "Multi/graph/kills_" + algname + str(index).replace("[", "").replace(", ", "").replace("]", "") + ".pdf"
    
    ylim = [0, 30]
    ep_avgx_mean_plot_multiple(kills, title = "Comparaison des ennemis tués pour " + algname + " sur VizDoom", algnames = algnames, path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond, ylabel = "Ennemis tués")

    #Ammos graph
    path = start_path + "Multi/graph/ammos_" + algname + str(index).replace("[", "").replace(", ", "").replace("]", "") + ".pdf"
    
    ep_avgx_mean_plot_multiple(ammos, title = "Comparaison des munititions pour " + algname + " sur VizDoom", algnames = algnames, path =  path, verbal = True, ylim=ylim, with_variance = True, n_run = repeat, stop_cond = stop_cond, win_cond = win_cond, ylabel = "Munitions utilisées")

def multi_graphs_reading_multiple(start_path, index = [0], lrs = [0.001, 0.00025], discounts = [0.999, 0.95], epsilons = [1], eps_decays = [0.9, 0.9999], eps_mins = [0.01], batch_sizes = [32], mems = [1000], warmups = [1000], repeat = 10, start_repeat = 0):
    '''
    Function for benchmarking for VizDoom that reads the data for multiple set of hyperparameters.
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
    sel_returns = []
    sel_steps = []
    sel_kills = []
    sel_ammos = []
    

    conf = config.Config()

    i = 0

    for mem in mems:
        for lr in lrs:
            for discount in discounts:
                for epsilon in epsilons:
                    for eps_decay in eps_decays:
                        for eps_min in eps_mins:
                            for batch_size in batch_sizes:
                                for warmup in warmups:
                                    if i in index:
                                        conf.lr_init = lr
                                        conf.discount = discount
                                        conf.epsilon = epsilon
                                        conf.epsilon_decay = eps_decay
                                        conf.epsilon_min = eps_min
                                        conf.batch_size = batch_size
                                        conf.warmup = warmup

                                        steps = []
                                        returns = []
                                        kills = []
                                        ammos = []
                                        for l in range(start_repeat, repeat):
                                            
                                            path = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/" + str(l) + ".pdf"
                                            path_csv = start_path + "mem"+str(mem)+"/lr"+str(lr)+"/disc"+str(discount)+"/eps"+str(epsilon)+"/eps_dec" + str(eps_decay) + "/b_s" + str(batch_size) + "/" +str(i) + "/csv/" + str(l)  + ".csv"

                                            tmp_step, tmp_returns, kills_tmp, ammos_tmp = read_csv(path_csv)
                                            steps.append(tmp_step)
                                            returns.append(tmp_returns)
                                            kills.append(kills_tmp)
                                            ammos.append(ammos_tmp)

                                        sel_steps.append(steps)
                                        sel_returns.append(returns)
                                        sel_kills.append(kills)
                                        sel_ammos.append(ammos)
                                    i += 1

    return sel_returns, sel_steps, sel_kills, sel_ammos #return the returns and steps for every repeat of every set of hyperparameters specified

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

    f(int(start), int(end), int(repeat), int(start_repeat))