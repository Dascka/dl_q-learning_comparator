import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import utils.csv_reading as csvr

def step_reward_plot(steps = [[]], 
                    returns = [[]], 
                    title = "",
                    alg_name = [''],
                    path = "./plot.pdf", 
                    verbal = True,
                    ylim = None, 
                    xlabel = "Episodes",
                    ylabel = "Retours"):
    '''

    :param steps: List of steps for each algorithms
    :param returns: List of returns for each algorithms according to the steps
    :param title: Title of the plot.
    :param alg_name: List of algorithm's names used.
    :param path: path where to save the plot
    :param verbal: True if we write the labels and the axes name. False otherwise.
    :param ylim: Set the y axis limits on the graph.
    :param xlabel: Set the x axis label on the graph.
    :param ylabel: Set the y axis label on the graph.
    ''' 
    plt.rcParams.update({'font.size': 36}) #To increase the size of the font on the labels

    fig, ax = plt.subplots(figsize = (20,20))

    colors = plt.cm.rainbow(np.linspace(0,1,len(steps))) #To have a flexible and sufficient amount of colors
    for i in range(len(steps)):
        x = steps[i]
        y = returns[i]
        ax.plot(x, y, color = colors[i], label = alg_name[i])
    
    if verbal:
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    # plt.show()
    plt.savefig(path)
    plt.close()

def ep_return_mean_plot(returns = [[]], 
                    title = "",
                    path = "./plot.pdf", 
                    verbal = True,
                    ylim = None, 
                    with_variance = False,
                    stop_cond = False,
                    win_cond = 195, 
                    xlim = None, 
                    xlabel = "Episodes",
                    ylabel = "Retours"):
    '''
    Compute for multiple runs the mean return obtained in function of the episode. Each list of return must have the same size : the number of episode played is the same for each run.
    :param returns: List of returns for each runs
    :param title: Title of the plot.
    :param path: path where to save the plot
    :param verbal: True if we write the labels and the axes name. False otherwise.
    :param ylim: Set the y axis limits on the graph.
    :param with_variance: True if we draw the variance on the graphs.
    :param stop_cond: True if there was a stopping condition on the training data we are aiming to plot.
    :param win_cond: Set the win condition of the game for which we are plotting.
    :param xlim: Set the x axis limits on the graph.
    :param xlabel: Set the x axis label on the graph.
    :param ylabel: Set the y axis label on the graph.
    '''
    plt.rcParams.update({'font.size': 36}) #To increase the size of the font on the labels

    fig, ax = plt.subplots(figsize = (20,20))

    if stop_cond:
        mean_y, std_y = tolerant_mean(returns)
        x = range(len(mean_y))
        ax.hlines(win_cond, xmin = xlim[0], xmax = xlim[1])
    else:
        mean_y = np.mean(returns, 0)
        std_y = np.std(returns, 0)
        x = range(len(mean_y))
        ax.hlines(win_cond, xmin = x[0], xmax = x[-1])

    colors = plt.cm.rainbow(np.linspace(0,1,len([0, 1])))

    y = mean_y
    ax.plot(x, y, color = colors[0])

    if with_variance:
        line1 = mean_y + std_y / np.sqrt(len(returns))
        line2 = mean_y - std_y / np.sqrt(len(returns))
        ax.plot(x, line1, "-", color = colors[0], alpha = 0.20)
        ax.plot(x, line2, "-", color = colors[0], alpha = 0.20)
        ax.fill_between(x, line1, line2, facecolor=colors[0], alpha=0.1, interpolate = True)
    
    if verbal:
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    if stop_cond and xlim is not None:
        ax.set_xlim(xlim)

    plt.savefig(path)
    plt.close()

    if stop_cond:
        run_completed(returns, path[:-4]+"_run_completed.pdf")

def ep_avgx_mean_plot(returns = [[]], 
                    title = "",
                    path = "./plot.pdf", 
                    verbal = True,
                    ylim = None, 
                    with_variance = False, 
                    n_run = 100,
                    stop_cond = False,
                    win_cond = 195,
                    xlim = None, 
                    xlabel = "Episodes",
                    ylabel = "Retours"):
    '''
    Compute for multiple runs the mean return obtained in function of the episode. Each list of return must have the same size : the number of episode played is the same for each run.
    :param returns: List of returns for each runs
    :param title: Title of the plot.
    :param path: path where to save the plot
    :param verbal: True if we write the labels and the axes name. False otherwise.
    :param ylim: Set the y axis limits on the graph.
    :param with_variance: True if we draw the variance on the graphs.
    :param n_run: The number of runs performed.
    :param stop_cond: True if there was a stopping condition on the training data we are aiming to plot.
    :param win_cond: Set the win condition of the game for which we are plotting.
    :param xlim: Set the x axis limits on the graph.
    :param xlabel: Set the x axis label on the graph.
    :param ylabel: Set the y axis label on the graph.
    '''
    plt.rcParams.update({'font.size': 36}) #To increase the size of the font on the labels

    fig, ax = plt.subplots(figsize = (20,20))

    y = []
    for l in returns:
        y.append(avg_x_episodes(l, n_run))

    if stop_cond:
        mean_y, std_y = tolerant_mean(y)
        x = range(len(mean_y))
        ax.hlines(win_cond, xmin = xlim[0], xmax = xlim[1])
    else:
        mean_y = np.mean(y, 0)
        std_y = np.std(y, 0)
        x = range(len(mean_y))
        ax.hlines(win_cond, xmin = x[0], xmax = x[-1])

    colors = plt.cm.rainbow(np.linspace(0,1,len([0, 1])))

    y = mean_y

    ax.plot(x, y, color = colors[0])

    if with_variance:
        line1 = mean_y + std_y / np.sqrt(len(returns))
        line2 = mean_y - std_y / np.sqrt(len(returns))
        ax.plot(x, line1, "-", color = colors[0], alpha = 0.20)
        ax.plot(x, line2, "-", color = colors[0], alpha = 0.20)
        ax.fill_between(x, line1, line2, facecolor=colors[0], alpha=0.1, interpolate = True)
    
    if verbal:
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

    # plt.show()
    plt.savefig(path)
    plt.close()

def avg_x_episodes(returns = [], n_run = 100):
    '''
    For a list of returns give the average return on the last n_run episodes for each episode.
    :param returns: The list of returns we are taking the mean over.
    :param n_run: The number of runs performed.
    '''
    if len(returns) < n_run:
        return []

    avg_n_run = []
    for i in range(1, len(returns)):
        tmp = max(0, i - n_run)
        avg_n_run.append(np.mean(returns[tmp: i]))

    return avg_n_run

def run_completed(returns = [[]], 
                    path = "./plot.pdf"):
    '''
    Compute for multiple runs the number of runs that are terminated in regards to the number of the episode.
    :param returns: List of returns for each runs
    :param path: path where to save the plot
    '''
    plt.rcParams.update({'font.size': 36}) #To increase the size of the font on the labels

    fig, ax = plt.subplots(figsize = (20,20))

    max_ep = max([len(x) for x in returns])
    run_completed = [0] * max_ep
    for l in returns:
        for i in range(len(l) - 1, max_ep):
            run_completed[i] += 1

    x = range(len(run_completed))

    colors = plt.cm.rainbow(np.linspace(0,1,len([0, 1])))
    y = run_completed
    ax.plot(x, y, color = colors[0])

    # plt.show()
    plt.savefig(path)
    plt.close()
        
def tolerant_mean(arrs):
    '''
    Compute the tolerant mean of arrs.
    https://stackoverflow.com/questions/10058227/calculating-mean-of-arrays-with-different-lengths
    :param arrs: List containing number and on which we are aiming to compute the tolerant mean.
    '''
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def ep_avgx_mean_plot_multiple(returns = [[[]]], 
                    title = "",
                    algnames = [""],
                    path = "./plot.pdf", 
                    verbal = True,
                    ylim = None, 
                    with_variance = False, 
                    n_run = 100,
                    stop_cond = False,
                    win_cond = 195,
                    xlim = None, 
                    xlabel = "Episodes",
                    ylabel = "Retours"):
    '''
    Compute for multiple runs the mean return obtained in function of the episode. Each list of return must have the same size : the number of episode played is the same for each run.
    :param returns: List of returns for each runs
    :param title: Title of the plot.
    :param algnames: The names of the algorithms.
    :param path: path where to save the plot
    :param verbal: True if we write the labels and the axes name. False otherwise.
    :param ylim: Set the y axis limits on the graph.
    :param with_variance: True if we draw the variance on the graphs.
    :param n_run: The number of runs performed.
    :param stop_cond: True if there was a stopping condition on the training data we are aiming to plot.
    :param win_cond: Set the win condition of the game for which we are plotting.
    :param xlim: Set the x axis limits on the graph.
    :param xlabel: Set the x axis label on the graph.
    :param ylabel: Set the y axis label on the graph.
    '''

    plt.rcParams.update({'font.size': 36}) #To increase the size of the font on the labels

    fig, ax = plt.subplots(figsize = (20,20))

    colors = plt.cm.rainbow(np.linspace(0,1,len(returns)))

    #store results until plot
    x_plot = []
    y_plot = []
    std_y_plot = []

    #for each algo
    for i in range(len(returns)):

        y = []
        for l in returns[i]:
            y.append(avg_x_episodes(l, n_run))

        if stop_cond:
            mean_y, std_y = tolerant_mean(y)
        else:
            mean_y = np.mean(y, 0)
            std_y = np.std(y, 0)

        x_plot.append(range(len(mean_y)))

        y_plot.append(mean_y)
        std_y_plot.append(std_y)

    x = None
    for i in range(len(x_plot)):
        if x is None or len(x_plot[i]) < len(x):
            x = x_plot[i]

    for i in range(len(y_plot)):
        y = y_plot[i][:len(x)]
        std_y = std_y_plot[i][:len(x)]

        ax.plot(x, y, color = colors[i], label = algnames[i])
        ax.hlines(win_cond, xmin = x[0], xmax = x[-1])

        if with_variance:
            line1 = y + std_y / np.sqrt(len(returns[i]))
            line2 = y - std_y / np.sqrt(len(returns[i]))
            ax.plot(x, line1, "-", color = colors[i], alpha = 0.20)
            ax.plot(x, line2, "-", color = colors[i], alpha = 0.20)
            ax.fill_between(x, line1, line2, facecolor=colors[i], alpha=0.1, interpolate = True)
    
    if verbal:
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

    # plt.show()
    plt.savefig(path)
    plt.close()