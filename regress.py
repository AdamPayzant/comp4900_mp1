import pandas as pd
import numpy as np
import os
import sys
from scipy.special import expit
from math import e

np.set_printoptions(threshold=sys.maxsize)

# class Regress:

#     def __init__(self, features):
#         self.weights = None
        
#     def fit(self, data, labels, lr, y, iterations=100):

#         if self.weights is None: self.weights = np.zeros(data.shape[1])

#         y = y.flatten()

#         for _ in range(iterations):
#             scores = np.dot(data, self.weights)
#             predictions = expit(scores)

#             error = y.T - predictions

#             gradient = np.dot(data.T, error)
#             self.weights += lr*gradient

#     def predict(self, x):
#         return expit(np.dot(x, self.weights))

#     def reset(self):
#         self.weights = None   

class Regress:

    def __init__(self, features):
        self.weights = None
        
    def fit(self, data, lr, y, iterations=100):

        # Initialize our weights
        if self.weights is None: self.weights = np.zeros(data.shape[1])

        # Flatten the classes
        y = y.flatten()

        # Run for the number of iterations
        for _ in range(iterations):
            # Find the gradient of our current model state
            grad = gradient(self.weights, data, y)
            # Update weights by the learning rate
            self.weights -= lr * grad

    def predict(self, x):
        # Return the models prediction for these features
        return int(model(x, self.weights) > 0.5)

    def reset(self):
        # Reset the models weights
        self.weights = None

def sigmoid(z):
    # Calculate the sigmoid value for the given input
    return 1 / (1 + np.exp(-z))

def model(x, weights):
    # Take the dot product of the input features and weights vectors
    # Then return the sigmoid of this result
    return sigmoid(np.dot(x, weights))

def gradient(weights, x, y):
    # Get the models predictions given these features
    preds = model(x, weights)
    # determine the gradient of these features
    # defined as: X (sigmoid(W_k^T dot X) - Y)
    gradient = np.dot(x.T, preds - y)
    return gradient