import os
import pandas as pd
import numpy as np
from numpy import random
import PyGnuplot as gp
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

def CreateCluster(N,k):
    random.seed(10)
    ppc = float(N)/k
    X = []
    for i in range(k):
        inc_centroid = random.uniform(20000.0, 200000.0)
        age_centroid = random.uniform(20.0, 70.0)
        for j in range(int(ppc)):
            X.append([np.random.normal(inc_centroid, 10000.0), np.random.normal(age_centroid, 2.0)])
    X = np.array(X)
    return X

data = CreateCluster(100, 5)

model = KMeans(n_clusters=5)

model = model.fit(scale(data))

print model.labels_

plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(np.float))
plt.show()
