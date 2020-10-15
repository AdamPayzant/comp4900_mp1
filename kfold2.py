import pandas as pd
import numpy as np
import os
import copy
import regress

class KFold2:
    def __init__(self, reg, k=10):
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

    def accuEval(self, lr, iterations, shuffled):
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


def loadCSV(filename):
    path = str(os.path.dirname(os.path.realpath(__file__)))      
    newPath = os.path.join(path, filename)
    return pd.read_csv(newPath)


lrVals = [0.01, 0.04, 0.08, 0.1, 0.2, 0.5, 0.8]
numIterations = [10, 40, 100, 200, 500, 1000, 5000]

df = loadCSV("bankrupcy.csv")
linReg = regress.Regress(len(df.iloc[:,:-1].columns))
kFoldData = KFold2(reg=linReg)
shuffled = kFoldData.shuffle(df)
for lr in lrVals:
    for iterations in numIterations:
        print("Average Error Rate: " + str(kFoldData.accuEval(lr=lr, iterations=iterations, shuffled=shuffled)))
        print("LR = " + str(lr) + "\tIterations = " + str(iterations))

bestAccuracyOne = 0
bestParametersOne = ""