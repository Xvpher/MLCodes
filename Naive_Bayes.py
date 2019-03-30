import os
import pandas as pd
import numpy as np
import PyGnuplot as gp
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import naive_bayes
import urllib2 as ulib

def data_loader():
    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    url_names = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"
    data_names = ulib.urlopen(url_names)
    names = []
    for i,line in enumerate(data_names):
        if i>32 and i<87:
            names.append(str(line.split()[0].split("_")[2].split(":")[0]))
        elif i>86:
            names.append(str(line.split()[0].split("_")[3].split(":")[0]))
    names.append("spam")
    df = pd.read_csv(url_data, header=None, names=names)
    df.to_csv("spamdata.csv", index=False)

def main():
    if not(os.path.exists("spamdata.csv")):
        data_loader()
    df = pd.read_csv("spamdata.csv")
    names = df.columns
    names = ['address','$','average']
    print df.loc[:,names]

if __name__ == '__main__':
    main()
