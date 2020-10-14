import pandas as pd
import numpy as np
import os
import copy
import regress

lrVals = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.5]
numIterations = [10, 40, 100, 200, 500, 1000, 4000, 10000, 25000]
results = {}

accuracyEval = {
    "accurate": 0,
    "inaccurate": 0
}

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
        if prediction == real:
            accuracyEval["accurate"] = accuracyEval["accurate"] + 1
        else:
            accuracyEval["inaccurate"] = accuracyEval["inaccurate"] + 1

def loadCSV(filename):
    path = str(os.path.dirname(os.path.realpath(__file__)))      
    newPath = os.path.join(path, filename)
    return pd.read_csv(newPath)

df = loadCSV("hepatitis.csv")
kFoldData = KFold()
shuffled = kFoldData.shuffle(df)
linReg = regress.Regress(len(df.iloc[:,:-1].columns))
for lr in lrVals:
    for iterations in numIterations:

        results[str(lr)+"-"+str(iterations)] = {
            "accuracy": [],
            "average": 0
        }

        for _ in range(5):

            accuracyEval["accurate"] = 0
            accuracyEval["inaccurate"] = 0

            for x in range(0, kFoldData.k):
                trainingSet, validationSet = kFoldData.splitAtX(kFoldData.k,x,shuffled)
                trainingSetData = trainingSet.iloc[:,:-1]
                classes = trainingSet.iloc[:,-1:]
                trainingSetLabel = trainingSet.iloc[:,-1:]
                linReg.fit(trainingSetData.to_numpy(), trainingSetLabel.to_numpy(), lr, classes.to_numpy(), iterations)
                validationSetData = validationSet.iloc[:,:-1]
                validationSetLabel = validationSet.iloc[:,-1:]
                validationSetLabel = validationSetLabel.to_numpy()
                count = 0
                for i in validationSetData.iterrows():
                    trainingArr = []
                    for j in range(0,len(i[1].to_numpy())):
                        trainingArr.append(i[1].to_numpy()[j])
                    kFoldData.accuEval(linReg.predict(trainingArr),validationSetLabel[count][0])
                    count += 1
                #print(accuracyEval)
            results[str(lr)+"-"+str(iterations)]["accuracy"].append( accuracyEval["accurate"]/(accuracyEval["accurate"]+accuracyEval["inaccurate"]))
        results[str(lr)+"-"+str(iterations)]["average"] = sum(results[str(lr)+"-"+str(iterations)]["accuracy"])/len(results[str(lr)+"-"+str(iterations)]["accuracy"])

        print(str(lr)+"-"+str(iterations) + ": " + str(results[str(lr)+"-"+str(iterations)]["average"]))

print(results)

bestAccuracy = 0
bestParameters = ""

for key, val in results.items():
    if val["average"] > bestAccuracy:
        bestAccuracy = val["average"]
        bestParameters = key
print(key+": "+str(bestAccuracy))