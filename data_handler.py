#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data Handling Module"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import numpy as np

def load_dataset(dataset_name):
    dataset = np.genfromtxt(dataset_name, dtype = "str", delimiter = ",")
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
    for instance in dataset:
        if instance[-1] == "positive":
            instance[-1] = 1
        else:
            instance[-1] = -1
    return dataset


def separate_attributes(dataset):
    dataset = {
        "input": dataset[:, 0:-1],
        "output": dataset[:, -1].astype(int)
    }
    return dataset


def save_results(data, file_name):
    for i in range(len(data)):
        with open(file_name + ".csv", "a") as f:
            f.write(str(i) + "," + str(data[i]) + "\n")
