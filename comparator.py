import benchmarks.Vizdoom.benchVizdoom as benchVizdoom
import benchmarks.OpenAI.benchCartPole as benchCartPole
import benchmarks.OpenAI.benchLunarLander as benchLunar
import argparse

def command_line_handler():
    """
    This function parses the arguments given in the command line using python's argparse module. A special format of command has been chosen and can be viewed by running "python comparator.py -h".
    :return: A Namespace containing the arguments and their values.
    """
    parser = argparse.ArgumentParser(description='A python implementation aiming to compare Deep Reinforcement Learning algorithms created from Q-learning')

    subparsers = parser.add_subparsers(title='Mode selection',
                                       description='This tool is used to launch the training of the differents algorithms and create the results graphs. It implements DQN, Double DQN, Dueling DQN and Dueling Double DQN for the games CartPole, LunarLander and VizDoom',
                                       help='Launch the training or create the graphs after a training', dest='mode')

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', help='Launch the training of an algorithme on a game. Some graphs are created during training, they are stored in the Result folder. The following path depends on the algorithm and hyperparameters.')

    # adding the game options (mutually exclusive and required)
    group_train = parser_train.add_mutually_exclusive_group(required=True)
    group_train.add_argument('-cp', action='store_true', dest='cp', help='Launch the training for CartPole.')

    group_train.add_argument('-ll', action='store_true', dest='ll', help='Launch the training for LunarLande.')

    group_train.add_argument('-vd', action='store_true', dest='vd', help='Launch the training for VizDoom.')

    parser_train.add_argument('-algo', required=True, type=str, action='store', dest='algo',
                              choices=['DQN', 'DDQN', 'Dueling', 'Dueling_DDQN', 'random'],
                              help='Choose the algorithm used between DQN, Double DQN (type DDQN), Dueling DQN (Type Dueling), Dueling Double DQN (type Dueling_DDQN) and random')
    parser_train.add_argument('-param', required=False, type=int, action='store', dest='param', nargs = 2,
                              metavar=('START_PARAM', 'END_PARAM'),
                              help='Launch the training with the hyperparameters set between START_PARAM and END_PARAM')
    parser_train.add_argument('-repeat', required=False, type=int, action='store', dest='repeat', nargs = 2,
                              metavar=('START_REPEAT', 'END_REPEAT'),
                              help='Perform the repeat of the training between START_REPEAT and END_REPEAT')

    # create the parser for the "graph" command
    parser_graph = subparsers.add_parser('bench', help='Draw the graphs for the selected game. The graphs will be stored in the Results folder. The following of the path depends on the algo and the hyperparameters')
    # adding the game options (mutually exclusive and required)
    group_bench = parser_graph.add_mutually_exclusive_group(required=True)
    group_bench.add_argument('-cp', action='store_true', dest='cp', help='Draw graphs for the CartPole game.')
    group_bench.add_argument('-ll', action='store_true', dest='ll', help='Draw graphs for the LunarLander game.')
    group_bench.add_argument('-vd', action='store_true', dest='vd', help='Draw graphs for the VizDoom game.')

    parser_graph.add_argument('-algo', required=True, type=str, action='store', dest='algo',
                              choices=['DQN', 'DDQN', 'Dueling', 'Dueling_DDQN', 'random', 'all'],
                              help='Choose the algorithm used between DQN, Double DQN (type DDQN), Dueling DQN (Type Dueling), Dueling Double DQN (type Dueling_DDQN), random and all')
    parser_graph.add_argument('-param', required=True, type=int, action='store', dest='param', nargs = 2,
                              metavar=('START_PARAM', 'END_PARAM'),
                              help='Launch the training with the hyperparameters set between START_PARAM and END_PARAM')
    parser_graph.add_argument('-repeat', required=True, type=int, action='store', dest='repeat', nargs = 2,
                              metavar=('START_REPEAT', 'END_REPEAT'),
                              help='Perform the repeat of the training between START_REPEAT and END_REPEAT')
    parser_graph.add_argument('-multi', required=False, action='store_true', dest='multi',
                                help='Choose whether the multi graphs are drawed.')

    return parser.parse_args()

def main():
    """
    Takes appropriate actions according to the chosen options (using command_line_handler() output).
    """

    # Parsing the command line arguments
    args = command_line_handler()

    #Retrieve the algo
    if args.algo == "DQN":
        algo = 0
    elif args.algo == "DDQN":
        algo = 1
    elif args.algo == "Dueling":
        algo = 2
    elif args.algo == "Dueling_DDQN":
        algo = 3
    elif args.algo == "random":
        algo = -1

    #Retrieve the hyperparameters
    start_param = args.param[0]
    end_param = args.param[1]

    #Retrieve the repeats
    start_repeat = args.repeat[0]
    end_repeat = args.repeat[1]

    if args.cp:
        bench = benchCartPole
    elif args.ll:
        bench = benchLunar
    elif args.vd:
        bench = benchVizdoom

    if args.mode == "train":
        """ ----- Training mode ----- """
        bench.train(start_param, end_param, end_repeat, start_repeat, algo)

    elif args.mode == "bench":
        """ ----- Graph mode ----- """

        var_all = args.algo == "all"
        multi = args.multi

        if var_all:
            bench.graphs(algo = 0)
            bench.graphs(algo = 1)
            bench.graphs(algo = 2)
            bench.graphs(algo = 3)
            bench.graphs(algo = -1)

            for DQN_n in range(start_param, end_param):
                for DDQN_n in range(start_param, end_param):
                    for Dueling_n in range(start_param, end_param):
                        for Dueling_DDQN_n in range(start_param, end_param):
                            bench.multi_graphs(DQN_n = DQN_n, DDQN_n=DDQN_n, Dueling_n=Dueling_n, Dueling_DDQN_n=Dueling_DDQN_n)
        else:
            bench.graphs(algo = algo, start_repeat=start_repeat, repeat = end_repeat)
            if args.multi:
                bench.multi_graphs_same_algo(index = list(range(start_param, end_param)), algo = algo, repeat = end_repeat, start_repeat = start_repeat)

main()