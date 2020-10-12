import pandas as pd
import numpy as np
import os
import copy
import regress

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
for x in range(0, kFoldData.k):
    trainingSet, validationSet = kFoldData.splitAtX(kFoldData.k,x,shuffled)
    trainingSetData = trainingSet.iloc[:,:-1]
    trainingSetLabel = trainingSet.iloc[:,-1:]
    linReg.fit(trainingSetData.to_numpy().tolist(),trainingSetLabel.to_numpy().tolist(),0.1,len(trainingSetData))
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
    print(accuracyEval)
