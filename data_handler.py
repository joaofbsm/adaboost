#!/usr/bin/env python3

"""Data Handling Module"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import numpy as np

def load_dataset(file_path):
    """Loads the dataset from a CSV like file.
    
    By using numpy's genfromtxt, imports the dataset contained in the file
    located in file_path.
    
    Arguments:
        file_path -- Path to the file that contains the dataset.
    """

    dataset = np.genfromtxt(file_path, dtype = "str", delimiter = ",")
    return dataset


def fold_dataset(dataset, k):
    """Create k folds on the dataset.
    
    Creates k folds for K-Fold Cross Validation.

    Arguments:
        dataset -- Dataset to be folded
        k -- Number of folds
    """

    np.random.shuffle(dataset)
    dataset = np.array_split(dataset, k)

    return dataset


def format_outputs(dataset):
    """Change output class name to corresponding integer.
    
    Even though the new labels are numbers, they are still stored as strings
    because of numpy's array limitation.
    
    Arguments:
        dataset -- Dataset represented as numpy array of arrays.
    """

    for instance in dataset:
        if instance[-1] == "positive":
            instance[-1] = 1
        else:
            instance[-1] = -1
    return dataset


def separate_attributes(dataset):
    """Separate inputs from outputs and return a dictionary.
    
    Now it is possible to convert outputs to true integers. 
    
    Arguments:
        dataset -- Dataset represented as numpy array of arrays.
    """

    dataset = {
        "input": dataset[:, 0:-1],
        "output": dataset[:, -1].astype(int)
    }
    return dataset


def save_results(data, file_path):
    """Save given list or array to CSV with indexes.
    
    Arguments:
        data -- List or array containing data.
        file_path -- Path to the file that will store the results.
    """

    # Empties a possible previous results file 
    f = open(file_path + ".csv", "w")
    f.write("i,e\n")  # Index, Error header
    for i in range(len(data)):
        # Saves row in CSV with indexes(iterations)
        with open(file_path + ".csv", "a") as f:
            f.write(str(i) + "," + str(data[i]) + "\n")
