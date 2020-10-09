import math
import pandas as pd
import numpy as np
import os

class regress:
    def __init__(self):
        self.w = np.array([])
        print("Initializing")
        
    def fit(self, data, labels):
        print("Fitting")
        if(self.w.size == 0):
            self.__estimateW(data)

    def __estimateW(self, data):    
        print("Estimating w")

    def predict(self, x):
        if(self.w.size == 0):
            return None
        if(__p0(data) > __p1(data)):
            return 0
        else:
            return 1

    def __p0(self, x):
        return -1/(1 + math.exp(- np.dot(w.T, x)))

    def __p1(self, x):
        return 1/(1 + math.exp(- np.dot(w.T, x)))

    def reset(self):
        self.w = np.array([])


"""
path = str(os.path.dirname(os.path.realpath(__file__)))

def calculateB(dataframe):
    x = dataframe.iloc[:,:-1]
    x['bias'] = 1
    xT = x.transpose()
    y = dataframe.iloc[:,-1:]
    xTy = xT.dot(y)
    xTimesxT = xT.dot(x)
    invXTimesxT = pd.DataFrame(np.linalg.pinv(xTimesxT.values), index=xTimesxT.columns)
    result = invXTimesxT.to_numpy().dot(xTy.to_numpy())
    return pd.DataFrame(result)

def calculateY(BHat, inputRow):
    tempB = BHat.to_numpy()
    total = 0
    for i in range(0,len(tempB)-1):
        total += tempB[i][0] * inputRow[i]
    total += tempB[len(tempB)-1]
    return total[0]

newPath = os.path.join(path, "hepatitis.csv")
df = pd.read_csv(newPath)
BHat = calculateB(df)
x = df.iloc[:,:-1]
results = []
for i in x.to_numpy():
    results.append(calculateY(BHat,i))
y = df.iloc[:,-1:].to_numpy()
for j in range(0,len(results)):
    print("y = ", results[j], "     actual result = ",y[j][0])
"""