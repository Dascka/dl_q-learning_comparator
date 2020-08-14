import pandas as pd
import numpy as np

class ListSizeError(Exception):
    def __init__(self, message):
        self.message = message

def read_from_one_csv(path = "./csv.csv"):
    '''
    Read the steps and according rewards from a unique csv file. The steps should be in first position and the reward in second.
    :param path: Path to the csv file.
    '''
    df = pd.read_csv(path).to_numpy()
    steps = df[:,0]
    returns = df[:,1]

    return steps, returns

def read_from_one_csv_vizdoom(path = "./csv.csv"):
    '''
    Read the steps and according rewards, kills and ammos from a unique csv file. The steps should be in first position and the reward in second.
    :param path: Path to the csv file.
    '''
    df = pd.read_csv(path).to_numpy()
    steps = df[:,0]
    returns = df[:,1]
    kills = df[:,2]
    ammos = df[:,3]

    return steps, returns, kills, ammos

def write_step_return_csv(steps = [], returns = [], path = "./step_return.csv"):
    '''
    Write a csv file with the given steps and the given returns. The two lists must have the same size.
    :param steps: List containing the steps.
    :param returns: List containing the returns.
    :param path: Path where to write the csv file.
    '''

    if len(steps) != len(returns):
        raise ListSizeError("Error, the size of the list steps and the list returns are not equals.")
        print("Error, the size of the list steps and the list returns are not equals.")
        return

    df = pd.DataFrame(list(zip(steps, returns)), columns=["Step", "Return"])
    df.to_csv(path, index = False)

def write_step_return_csv_vizdoom(steps = [], returns = [], kills = [], ammos = [], path = "./step_return.csv"):
    '''
    Write a csv file with the given steps, the given returns, the kills and the ammo used. The lists must have the same size.
    :param steps: List containing the steps.
    :param returns: List containing the returns.
    :param kills: List containing the kills.
    :param ammos: List containing the remaining ammos.
    :param path: Path where to write the csv file.
    '''

    if len(steps) != len(returns):
        raise ListSizeError("Error, the size of the list steps and the list returns are not equals.")
        print("Error, the size of the list steps and the list returns are not equals.")
        return

    df = pd.DataFrame(list(zip(steps, returns, kills, ammos)), columns=["Step", "Return", "Kills", "Ammo used"])
    df.to_csv(path, index = False)