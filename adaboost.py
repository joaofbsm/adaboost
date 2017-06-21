#!/usr/bin/env python3

"""AdaBoost Class Structure"""

__author__ = "João Francisco B. S. Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "GPL"
__version__ = "3.0"

import sys
import numpy as np
import time

class AdaBoost:
    """Implementation of AdaBoost boosting method.
    
    AdaBoost combines weak learners to create a strong learning hypothesis.
    Our weak learners are essentially 1-level decision trees, commonly known as 
    decision stumps.
    """

    def __init__(self, training_set, testing_set):
        """Initialize AdaBoost object.
        
        Arguments:
            training_set -- Dataset for training in.
            testing_set --  Dataset for testing the hypotheses.
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


    def evaluate_stump(self, stump):
        """Returns the stump error in current weighted training set.
        
        Arguments:
            stump -- Stump to be evaluated.
        """

        predictions = np.zeros(self.m_tr)  # Hypothesis for each instance
        pred_errors = np.ones(self.m_tr)  # 0 if correct, 1 if incorrect
        a = stump["attribute"]  # Attribute index in training set
        # Loop through instances
        for i in range(self.m_tr):
            value = self.training_set["input"][i][a]
            output = self.training_set["output"][i]
            if value == stump["value"]:
                predictions[i] = stump["state"]
            else:
                predictions[i] = stump["state"] * -1
            if predictions[i] == output:
                pred_errors[i] = 0

        # Should divide by the sum of the weights, but it is always 1
        error = np.sum(np.multiply(self.weights, pred_errors))

        return error, predictions


    def find_best_stump(self):
        """Return the best decision stump for current weights.

        Creates 54 different decision stumps(9 attributes * 3 possible values
        * 2 states[True or False] combinations). 
        """

        best_stump = {}
        lowest_error = float("inf")
        possible_values = ["x", "o", "b"]
        possible_states = [1, -1]
        # Loop through attributes
        for a in range(self.n_tr):
            for value in possible_values:
                for state in possible_states:
                    # Instantiates stump
                    stump = {"attribute": a}
                    stump["value"] = value
                    # Predict this for value or -1 * this for not value
                    stump["state"] = state  

                    # Calculate error for stump
                    error, predictions = self.evaluate_stump(stump)
                    stump["error"] = error
                    stump["predictions"] = predictions

                    if error < lowest_error:
                        lowest_error = error
                        best_stump = stump

        return best_stump


    def calculate_alpha(self, model):
        """Calculates alpha for the error of the given(best) model.
        
        Attributes:
            model = Best predicting weak learner for time t.
        """
        error = model["error"]
        alpha = 0.5 * np.log((1 - error) / error)
        
        return alpha


    def update_weights(self, model, alpha):
        """Update weights for time t according to AdaBoost's formula.

        Attributes:
            model = Best predicting weak learner for time t.
            alpha = Alpha calculated for model.

        Equivalent implementations of weight calculation
        -------------------------------------------------

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


    def evaluate_ensemble(self):
        """Evaluate current strong learner with the testing set."""

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
                if value == self.ensemble[model]["value"]:
                    prediction = self.ensemble[model]["state"]
                else:
                    prediction = self.ensemble[model]["state"] * -1
                H += self.alpha[model] * prediction
            H = np.sign(H)  # Strong model hypothesis

            if H == self.testing_set["output"][i]:
                correct += 1

        accuracy = (correct / self.m_ts) * 100  # Simple accuracy measure
        error = 100 - accuracy

        return accuracy, error


    def boost(self, num_iterations):
        """The AdaBoost algorithm itself.
        
        Uses all the above methods together to boost the best weak learners 
        created in every iteration by combining them into a strong learner
        that gets better over time.
        
        Arguments:
            num_iterations -- Number of iterations in the process of boosting.
        """

        accuracies = []  # Accuracy per iteration
        errors = []  # Error per iteration
        model_errors = []  # Errors for the best model in each iteration
        for i in range(num_iterations):
            best_model = self.find_best_stump()
            model_errors.append(best_model["error"] * 100)
            self.ensemble.append(best_model)
            self.alpha.append(self.calculate_alpha(best_model)) 

            results = self.evaluate_ensemble()
            accuracies.append(results[0])
            errors.append(results[1])

            self.update_weights(best_model, self.alpha[i])
        return accuracies, errors, model_errors
