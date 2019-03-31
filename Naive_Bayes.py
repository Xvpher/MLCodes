import os
import pandas as pd
import numpy as np
import PyGnuplot as gp
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
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
    names = names[0:len(names)-1]
    X = df.loc[:,names].values
    y = df.loc[:,'spam'].values
    X,y = shuffle(X,y)
    X_train = X[:3680,:]
    y_train = y[:3680]
    X_test = X[3680:,:]
    y_test = y[3680:]
    model = MultinomialNB()
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    # for i in range(100):
    #     print "{} \t {} \n".format(pred[i], y_test[i])
    # print model.predict_proba(X_test)
    print model.score(X_test,y_test)

if __name__ == '__main__':
    main()
