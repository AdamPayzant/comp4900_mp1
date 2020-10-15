import pandas as pd
import numpy as np
import os
import copy
import regress

class KFold2:
    def __init__(self, k, reg):
        self.k = k
        self.reg = reg
    
    def shuffle(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def splitAtX(self, k,x, df):
        sizeOfSets = len(df) // k
        training = copy.deepcopy(df)
        validation = training[sizeOfSets*x:sizeOfSets+(sizeOfSets*x)]
        for i in range(sizeOfSets*x,sizeOfSets+(sizeOfSets*x)):
            training = training.drop(i)
        self.training = training
        self.validatation = validation
        return training, validation

    def accuEval(self, lr, iterations, trainingSet, shuffled):
        err = 0
        accurate = 0
        inaccurate = 0

        for _ in range(5):
            count = 0
            accurate = 0
            inaccurate = 0

            for x in range(0, self.k):
                trainingSet, validationSet = self.splitAtX(self.k,x,shuffled)
                trainingSetData = trainingSet.iloc[:,:-1]
                classes = trainingSet.iloc[:,-1:]
                self.reg.fit(trainingSetData.to_numpy(), lr, classes.to_numpy(), iterations)
                validationSetData = validationSet.iloc[:,:-1]
                validationSetLabel = validationSet.iloc[:,-1:]
                validationSetLabel = validationSetLabel.to_numpy()
                count = 0
                for i in validationSetData.iterrows():
                    trainingArr = []
                    for j in range(0,len(i[1].to_numpy())):
                        trainingArr.append(i[1].to_numpy()[j])
                    if self.reg.predict(trainingArr) == validationSetLabel[count][0]:
                        accurate += 1
                    else:
                        inaccurate += 1
                    count += 1
                err += inaccurate/count
                #print(accuracyEval)
            err = err/5
        return err

    def run(self):
        return 0