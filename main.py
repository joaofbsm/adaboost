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
import data_handler as dh

def main():
    k = 5  # Number of folds

    # Dataset retrieving and formatting
    dataset = dh.load_dataset("tic-tac-toe.data")
    dataset = dh.format_outputs(dataset)
    dataset = dh.fold_dataset(dataset, k)

    cv_accuracies = []
    cv_errors = []
    for i in range(k):
        print("Round:", i, "\n")
        testing_set = dh.separate_attributes(dataset[i])
        remaining_folds = np.concatenate(np.delete(dataset, i))
        training_set = dh.separate_attributes(remaining_folds)

        ada = ab.AdaBoost(training_set, testing_set)
        results = ada.boost(500)

        cv_accuracies.append(results[0])
        cv_errors.append(results[1])

    # Convert lists to numpy arrays for faster calculations
    cv_accuracies = np.asarray(cv_accuracies)
    cv_errors = np.asarray(cv_errors)

    # Calculate the mean of the accuracies and the errors
    cv_accuracies = np.divide(np.sum(cv_accuracies, axis=0), k)
    cv_errors = np.divide(np.sum(cv_errors, axis=0), k)

    # Save the results to a CSV
    dh.save_results(cv_accuracies, "boosting_accuracy")
    dh.save_results(cv_errors, "boosting_error")

if __name__ == "__main__":
    main()
