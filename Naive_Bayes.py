import os
import pandas as pd
import numpy as np
import PyGnuplot as gp
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import naive_bayes
import urllib2 as ulib

url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
url_names = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"
data_names = ulib.urlopen(url_names)
names = []
for i,line in enumerate(data_names):
    if i>32 and i<87:
        names.append(str(line.split()[0].split("_")[2].split(":")[0]))
    elif i>86:
        names.append(str(line.split()[0].split("_")[3].split(":")[0]))
df = pd.DataFrame(columns = names)
data = ulib.urlopen(url_data)
for line in data:
    value = [x.rstrip() for x in line.split(",")]
    # print np.array(value)
    df.append(value)
    break
print df.shape    
