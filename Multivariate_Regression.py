import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import PyGnuplot as gp
from scipy import stats
import urllib2 as ulib
from matplotlib import pyplot as plt

url_names = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names"
url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
data = ulib.urlopen(url_names)
names = list()
for i,line in enumerate(data):
    if i>32 and i<42:
        line = line.strip()
        j=3
        while(line[j]!=':'):
            j+=1
        names.append(str(line[3:j]))
df_data = {}
for i in range(9):
    df_data[names[i]] = []
data = ulib.urlopen(url_data)
for line in data:
    values = line.split()
    for i in range(8):
        if(values[i] == '?'): values[i]=-999
        df_data[names[i]].append(float(values[i]))
    df_data[names[8]].append(" ".join(values[8:]))
df = pd.DataFrame(df_data)
df = df[df['horsepower'] != -999]
used_f = 2
X = df.loc[:, ['acceleration','horsepower']]
Y = df.loc[:, ['mpg']].values
X = (X - X.mean())/X.std()
Y = (Y - Y.mean())/Y.std()
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X), axis=1)
theta = np.zeros([1,used_f+1])
alpha = 0.01
iters = 100

def computeCost(X,Y,theta):
    h = np.power(((np.matmul(X,np.transpose(theta)))-Y),2)
    s = np.sum(h)/(2*len(X))
    return s

x_cord = np.empty(iters)
y_cord = np.empty(iters)

def gradientDescent(X,Y,theta,alpha,iters):
    cost = np.zeros(iters)
    for i in range(iters):
        theta -= (alpha/len(X)) * (np.matmul(np.transpose((np.matmul(X,np.transpose(theta)) - Y)), X))
        x_cord[i] = theta[0][1]
        y_cord[i] = theta[0][2]
        cost[i] = computeCost(X, Y, theta)
    return theta,cost

g,cost = gradientDescent(X,Y,theta,alpha,iters)
# print g
scale(x_cord)
scale(y_cord)
scale(cost)
finalCost = computeCost(X,Y,g)
# print(finalCost)

gp.c('set term pdfcairo')
gp.c('set output "output_2.pdf"')

fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

gp.s([x_cord,y_cord,cost], filename='3d_plot_data.dat')
gp.c('set key off')
gp.c('splot "3d_plot_data.dat" using 1:2:3 with points palette pointsize 3 pointtype 7')

gp.c('set view map')
gp.c('set size ratio .9')
gp.c('set object 1 rect from graph 0, graph 0 to graph 1, graph 1 back')
gp.c('set object 1 rect fc rgb "black" fillstyle solid 1.0')
gp.c('splot "3d_plot_data.dat" using 1:2:3 with points pointtype 5 pointsize 1 palette linewidth 30')
