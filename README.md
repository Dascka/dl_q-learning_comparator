# Comparaison of Q-Learning algorithms tested on CartPole, LunarLander and [VizDoom](http://vizdoom.cs.put.edu.pl/).

-----

Python 3.7.4 implementation of a comparator of Q-Learning algorithms such as DQN, Double DQN, Dueling DQN and Dueling Double DQN.
The bases of the agents implementation come from https://github.com/nilportugues/reinforcement-learning-1 for the CartPole game and from https://github.com/flyyufelix/VizDoom-Keras-RL for the [VizDoom](http://vizdoom.cs.put.edu.pl/) game. Every agent implementation is adapted to work well within this project.

A project created by Dorian Labeeuw, master student, University of Mons, for his master thesis.

-----

## Description

This project allows the user to train the four algorithm and draw graphs with the corresponding results. This project attends to present good results in term of reproductability and quality like its presented in https://arxiv.org/pdf/1709.06560.pdf. If someone wants to consult the report of this project explaining the results obtained, feel free to contact the author.

The results are not stored in this repository because of too many pdf and csv files.
-----

## Dependencies

- keras==2.0.5
- tensorflow==1.2.1
- vizdoom
- scikit-image
- pandas
- gym

To install the dependencies :
```
pip install -r requirements.txt
```

-----

## Structure
    .
    ├── comparator.py   #Console user interface (allowing to train an algorithm or draw the graphs)
    ├── agents          #Agents implementation
    ├── benchmarks      #Benchmarking functions used for performance testing
    ├── config          #Config files used to store the hyperparameters
    ├── Results         #Folder where the results are stored
    ├── utils           #Utilities files such as the graph drawer and the csv reader
    └── VizDoom         #[VizDoom](http://vizdoom.cs.put.edu.pl/) necessary files

-----

## How to run

- To show help dialog : comparator.py -h

- To launch the training of DQN for CartPole : comparator.py train -cp -algo DQN -param 0 1 -repeat 0 10

This will launch the training of the algorithm DQN for the game CartPole. 

-cp can be replaced with -ll for LunarLander or -vd for VizDoom.

-algo define the algorithm used. It is: random, DQN, DDQN, Dueling or Dueling_DDQN.

-param X Y precises the set of hyperparamers to use. See the benchmark file associated to the game to know the different set of hyperparameters. To precise the parameters, the user has to define the starting point X and the ending point Y in the different set of hyperparameters.

-repeat X Y precises the number of repeat to perform. A repeat is the number of time a training is performed with the same algorithm, the same game and the same hyperparameters but with a different random seed. The user has to define the starting point of the repeat X and the ending point of the repeat T.

- To launch the drawing of graphs from datas obtained from a trained algorithm : comparator.py bench -cp -algo all -param 0 1 - repeat 0 10 -multi

It draws the graphs for every algorithm for the game CartPole. It takes the same argument as we have seen before but:

-algo can take the option all.

-multi define if the tool must draw graphs with data from every algo on the same graph.