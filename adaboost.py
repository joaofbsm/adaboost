#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AdaBoost Class Structure"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import numpy as np

class AdaBoost:
    """Implementation of AdaBoost boosting method.
    
    AdaBoost combines weak learners to create a strong learning hypothesis.
    In this code we use the one rule method to get the best weak learner at
    each iteration. Our weak learners are essentially 1-level decision trees,
    commonly known as decision stumps.
    """

    def __init__(self, training_set, testing_set):
        """Initialize AdaBoost object.
        
        Arguments:
            training_set -- Dataset for training in.
            testing_set --  Dataset for testing the hypotheses
        """
        self.training_set = training_set
        self.testing_set = testing_set
         # Number of training instances
        self.m_tr = training_set["input"].shape[0] 
        # Number of input attributes(The same for testing)
        self.n_tr = training_set["input"].shape[1]  
        # Number of testing instances
        self.m_ts = testing_set["input"].shape[0]
        # Weights for training instances
        self.weights = np.divide(np.ones(self.m_tr), self.m_tr)  
        # Collection of chosen weak learners
        self.ensemble = [] 
        # Weight assigned to each weak learner
        self.alpha = []  


    def stump_error(self, stump):
        predictions = np.zeros(self.m_tr)
        pred_errors = np.ones(self.m_tr)  # 0 if correct, 1 if incorrect
        a = stump["attribute"]  # Attribute index in training set
        # Loop through instances
        for i in range(self.m_tr):
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
        for a in range(self.n_tr):
            stump = {"attribute": a}

            # Ocurrences of output class times instance weight per input value
            ocurrences = {
                "x": np.zeros(2),
                "o": np.zeros(2),
                "b": np.zeros(2)
            }
            # Loop through instances
            for i in range(self.m_tr):
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

        for i in range(self.m_tr):
            self.weights[i] = self.weights[i] * np.exp(-1 * alpha * 
                      self.training_set["output"][i] * model["predictions"][i])

        self.weights = np.divide(self.weights, np.sum(self.weights))
        -----------------------------------------------------------------------
        for i in range(self.m_tr):
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
        for i in range(self.m_ts):
            H = 0
            for model in range(len(self.ensemble)):
                # Get the attribute that the model is related with
                a = self.ensemble[model]["attribute"]
                # Get the value(class) it presents in this instance
                value = self.testing_set["input"][i][a]
                # Predict according to model rules
                prediction = self.ensemble[model][value]
                H += self.alpha[model] * prediction
            H = np.sign(H)

            if H == self.testing_set["output"][i]:
                correct += 1

        accuracy = (correct / self.m_ts) * 100
        error = 100 - accuracy

        return accuracy, error


    def boost(self, num_iterations):
        accuracies = []  # Accuracy per iteration
        errors = []  # Error per iteration
        for i in range(num_iterations):
            best_model = self.one_rule()
            self.ensemble.append(best_model)
            self.alpha.append(self.calculate_alpha(best_model)) 

            results = self.evaluate()
            accuracies.append(results[0])
            errors.append(results[1])
            
            self.update_weights(best_model, self.alpha[i])
            print("Iteration:", i)
        print("The accuracy for the final classifier is: ", results[0], "%", 
          sep="")
        print("The error for the final classifier is: ", results[1], "%\n", 
          sep="")
        return accuracies, errors

