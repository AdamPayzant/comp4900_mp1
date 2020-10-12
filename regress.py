import pandas as pd
import numpy as np
import os
import sys
from scipy.special import expit

np.set_printoptions(threshold=sys.maxsize)

class Regress:
    # I hate this class and hope it will burn. I'm uncertain about everything in my implementation
    def __init__(self, features):
        # features - The number of features in the model
        self.w = np.zeros((1, features))
        self.features = features

    def __sigma(self, x):
        return 1/(1 + expit(-(np.dot(x, self.w.T))))
        
    def fit(self, data, labels, lr, iterations=100):
        # data - The data to train on
        # labels - The labels for the training data
        # lr - The learning rate
        # iterations - The number of iterations to learn with
        a = -lr
        wk1 = np.zeros((1, self.features))
        for _ in range(0,iterations):
            a += lr
            sum = 0
            for i in range(0, self.features):
                sum += data[i] * (labels[0] - self.__sigma(data[i]))
            wk1 = self.w - a * sum
            # TODO: CONDITIONAL FOR THE END CASE
            # w_k+1 - w_k || < e
            #     end
            self.w = wk1

    def predict(self, x):
        # x - the data to predict from
        # Compares P(x=1 | Y) and P(x=0 | Y)
        if(self.__p0(x) > self.__p1(x)):
            return 0
        else:
            return 1

    def __p0(self, x):
        # Finds P(x=0 | Y)
        return 1 - self.__sigma(x)
    def __p1(self, x):
        # Finds P(x=1 | Y)
        return self.__sigma(x)
    def reset(self):
        # Resets the model, I don't know why you would need this, but it's there?
        self.w = np.array([])
