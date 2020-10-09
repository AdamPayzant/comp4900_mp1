import pandas as pd
import numpy as np
import os
import copy

class KFold:
    def __init__(self, k=10):
        self.k = k

    def shuffle(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def splitAtX(self, k,x, df):
        sizeOfSets = len(df) // k
        training = copy.deepcopy(df)
        validation = training[sizeOfSets*x:sizeOfSets+(sizeOfSets*x)]
        for i in range(sizeOfSets*x,sizeOfSets+(sizeOfSets*x)):
            training = training.drop(i)
        return training, validation

    def accuEval(self, prediction, real):
        print("Evaluating")

def loadCSV(filename):
    path = str(os.path.dirname(os.path.realpath(__file__)))      
    newPath = os.path.join(path, filename)
    return pd.read_csv(newPath)
