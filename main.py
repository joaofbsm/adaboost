#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AdaBoost binary classification from scratch in Python"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import numpy as np
import adaboost as ab


def load_dataset(dataset_name):
    dataset = np.genfromtxt(dataset_name, dtype = "str", delimiter = ",")
    return dataset

def output_to_integers(dataset):
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

def main():
    # Dataset retrieving and formatting
    dataset = load_dataset("tic-tac-toe.data")
    dataset = output_to_integers(dataset)
    dataset = separate_attributes(dataset)

    ada = ab.AdaBoost(dataset)

    ada.boost(500)


if __name__ == "__main__":
    main()
