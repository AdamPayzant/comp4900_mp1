import math
import pandas as pd
import numpy as np
import os

class Regress:
    # I hate this class and hope it will burn. I'm uncertain about everything in my implementation
    def __init__(self, features):
        self.w = np.zeros((1, features))
        self.features = features
        
    def fit(self, data, labels, lr):
        a = -lr
        wk1 = np.zeros((1, self.features))
        # TODO: FIX THE RANGE
        # OTherwise I think this is right?
        for _ in range(0,100):
            a += lr
            sum = 0
            for i in range(0, self.features):
                sum += data[i] * (labels[i] - (1 / (1 + math.exp(- np.dot(self.w.T, data[i])))))
            wk1 = self.w - a * sum
            # TODO: CONDITIONAL FOR THE END CASE
            # w_k+1 - w_k || < e
            #     end
            self.w = wk1

    def predict(self, x):
        # Compares P(x=1 | Y) and P(x=0 | Y)
        if(self.__p0(x) > self.__p1(x)):
            return 0
        else:
            return 1

    def __p0(self, x):
        # Finds P(x=0 | Y)
        return -1/(1 + math.exp(- np.dot(self.w.T, x)))
    def __p1(self, x):
        # Finds P(x=1 | Y)
        return 1/(1 + math.exp(- np.dot(self.w.T, x)))
    def reset(self):
        # Resets the model, I don't know why you would need this, but it's there?
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