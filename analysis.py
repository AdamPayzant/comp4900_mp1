import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


def loadCSV(filename):
    path = str(os.path.dirname(os.path.realpath(__file__)))      
    newPath = os.path.join(path, filename)
    return pd.read_csv(newPath)

hepatitisData = loadCSV("hepatitis.csv")
bankruptcyData = loadCSV("bankrupcy.csv")

numClasses = 20

def generateHistogramData(dataset, datasetName):
    collected = {}
    for data in dataset:
        collected[data] = {
            "rawData": [],
            "step": 1
        }
        for i in range(0, len(dataset[data])):
            collected[data]["rawData"].append(dataset[data][i])

        collected[data]["step"] = max(collected[data]["rawData"]) - min(collected[data]["rawData"])

    for entry in collected:
        plt.title(entry+" distribution")
        plt.xlabel(entry)
        plt.ylabel("occurences")
        if(collected[entry]["step"] == 1):
            plt.hist(collected[entry]["rawData"], bins=2)
        else:
            plt.hist(collected[entry]["rawData"], bins=numClasses)
        
        if(not os.path.exists("plots/")): os.mkdir("plots/")
        if(not os.path.exists("plots/"+str(datasetName)+"/")): os.mkdir("plots/"+str(datasetName)+"/")
        plt.savefig("plots/"+str(datasetName)+"/"+str(entry)+"-distribution.png")
        plt.close()

generateHistogramData(hepatitisData, "hepatitis")
generateHistogramData(bankruptcyData, "bankruptcy")