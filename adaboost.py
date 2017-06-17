#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AdaBoost Class Structure"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import time
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
        predictions = np.zeros(self.m)
        pred_errors = np.ones(self.m)  # 0 if correct, 1 if incorrect
        a = stump["attribute"]  # Attribute index in training set
        # Loop through instances
        for i in range(self.m):
            value = self.training_set["input"][i][a] 
            output = self.training_set["output"][i]
            predictions[i] = stump[value]
            if predictions[i] == output:
                pred_errors[i] = 0

        error = np.sum(np.multiply(self.weights, pred_errors))

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
                    ocurrences[key][0] += self.weights[i]
                else:  # Output is "negative"
                    ocurrences[key][1] += self.weights[i]

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
                lowest_error = error
                best_stump = stump

        return best_stump


    def calculate_alpha(self, model):
        error = model["error"]
        alpha = 0.5 * np.log((1 - error) / error)
        
        return alpha


    def update_weights(self, model, alpha):
        """
        Equivalent implementations of weight calculation:

        for i in range(self.m):
            self.weights[i] = self.weights[i] * np.exp(-1 * alpha * 
                      self.training_set["output"][i] * model["predictions"][i])

        self.weights = np.divide(self.weights, np.sum(self.weights))
        -----------------------------------------------------------------------
        for i in range(self.m):
            if model["predictions"][i] != self.training_set["output"][i]:
                self.weights[i] = np.divide(self.weights[i], 
                                            2 * model["error"])
            else:
                self.weights[i] = np.divide(self.weights[i], 
                                            2 * (1 - model["error"]))
        """ 

        self.weights = np.multiply(self.weights, 
                                   np.exp(-1 * alpha 
                                     * np.multiply(self.training_set["output"],
                                                   model["predictions"])
                                   )
                       )
        
        self.weights = np.divide(self.weights, np.sum(self.weights))


    def evaluate(self):
        correct = 0
        # Loop through instances
        for i in range(self.m):
            H = 0
            for model in range(len(self.ensemble)):
                H += self.alpha[model] * self.ensemble[model]["predictions"][i]
            H = np.sign(H)

            if H == self.training_set["output"][i]:
                correct += 1

        accuracy = (correct / self.m) * 100

        print("The accuracy for the final classifier is: ", accuracy, "%", 
              sep="")


    def boost(self, num_iterations):
        for i in range(num_iterations):
            best_model = self.one_rule()
            self.ensemble.append(best_model)
            self.alpha.append(self.calculate_alpha(best_model)) 
            self.evaluate()
            self.update_weights(best_model, self.alpha[i])

