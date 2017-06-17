#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AdaBoost Class Structure"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import numpy as np

# TODO
# - Convert stump to class
#

class AdaBoost:
    """Implementation of AdaBoost boosting method.
    
    AdaBoost combines weak learners to create a strong learning hypothesis.
    In this code we use the one rule method to get the best weak learner at
    each iteration. Our weak learners are essentially 1-level decision trees,
    commonly known as decision stumps.
    """

    def __init__(self, training_set):
        """Initialize AdaBoost object.
        
        Arguments:
            training_set -- Dataset for training in.
        """
        self.training_set = training_set
        self.m = training_set["input"].shape[0]  # Number of instances
        self.n = training_set["input"].shape[1]  # Number of input attributes
        self.weights = np.divide(np.ones(self.m), self.m)  # Instance weights
        self.ensemble = []  # Collection of chosen weak learners
        self.alpha = []  # Weight assigned to each weak learner

    def stump_error(self, stump):
        predictions = np.ones(self.m)  # 0 if correct, 1 if incorrect
        a = stump["attribute"]  # Attribute index in training set
        # Loop through instances
        for i in range(self.m):
            value = self.training_set["input"][i][a] 
            output = self.training_set["output"][i]
            if stump[value] == output:
                predictions[i] = 0

        error = np.divide(np.sum(np.multiply(self.weights, predictions)), 
                          np.sum(self.weights))

        return error, predictions


    def one_rule(self):
        """Return the best decision stump for current weights."""

        best_stump = {}
        lowest_error = float("inf")

        # Loop through attributes
        for a in range(self.n):
            stump = {"attribute": a}

            # Ocurrences of output class per input value
            ocurrences = {
                "x": np.zeros(2),
                "o": np.zeros(2),
                "b": np.zeros(2)
            }
            # Loop through instances
            for i in range(self.m):
                key = self.training_set["input"][i][a]
                if self.training_set["output"][i] == 1:  # Output is "positive"
                    ocurrences[key][0] += 1
                else:  # Output is "negative"
                    ocurrences[key][1] += 1
            
            # Create rule based in most frequent output class per input value
            for key in ocurrences:
                if ocurrences[key][0] >= ocurrences[key][1]:
                    stump[key] = 1  # Value predicts "positive" output
                else:
                    stump[key] = -1  # Value predicts "negative" output

            error, predictions = self.stump_error(stump)
            stump["error"] = error
            stump["predictions"] = predictions
            if error < lowest_error:
                best_stump = stump

        return best_stump
    

    def calculate_alpha(self, model):
        error = model["error"]
        alpha = 0.5 * np.log((1 - error) / error)
        
        return alpha


    def update_weights(self, model, alpha):
        weights_sum = np.sum(self.weights)

        # Here y_i * h_t(x_i) is equivalent to the model predictions
        self.weights = np.divide(
                                 np.multiply(self.weights, 
                                             np.exp(
                                                    np.multiply(
                                                       alpha, 
                                                       model["predictions"]
                                                    )
                                             )
                                 ), 
                                 weights_sum
                       )


    def boost(self, num_iterations):
        for i in range(num_iterations):
            best_model = self.one_rule()
            print(best_model["error"])
            self.ensemble.append(best_model)
            self.alpha.append(self.calculate_alpha(best_model)) 
            self.update_weights(best_model, self.alpha[i])
