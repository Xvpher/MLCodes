import numpy as np
import pandas as pd
import PyGnuplot as gp
from scipy import stats
from sklearn.metrics import r2_score

np.random.seed(2)
mu, sigma, n = 3.0, 1.0, 1000
x = np.random.normal(mu, sigma, n)
# y = 100 - (x + np.random.normal(0.0, 0.5, 1000))*3
y = np.random.normal(mu, sigma, n) / x**3
data = {'X':x, 'Y':y}
df = pd.DataFrame(data)
u_quart = np.percentile(y,75)
l_quart = np.percentile(y,25)
iqr = (u_quart - l_quart)*2
value = (l_quart - iqr, u_quart + iqr)
good_val = df[(df['Y']>=value[0]) & (df['Y']<=value[1])]
df = good_val
l = int(0.8*df.shape[0])
x = np.array(df['X'])
y = np.array(df['Y'])
xtrain = np.array(df['X'][:l])
xtest = np.array(df['X'][l:])
ytrain = np.array(df['Y'][:l])
ytest = np.array(df['Y'][l:])


slope, intercept, rval, pval, stderr = stats.linregress(xtrain,ytrain)
lin = slope*xtrain + intercept
ypred = slope*xtest + intercept

p = np.poly1d(np.polyfit(xtrain, ytrain, 8))
xp = np.linspace(1,7,100)

gp.c('set term pdfcairo')
gp.c('set output "output.pdf"')
gp.s([x,y], filename='plot_data.dat')
gp.s([xtrain,ytrain], filename='plot_data1.dat')
gp.s([xtest,ytest], filename='plot_data2.dat')
gp.s([xtrain,lin], filename='plot_data3.dat')
gp.s([xp, p(xp)], filename='plot_data4.dat')

gp.c('set style fill solid 0.5 border -1')
gp.c('set style boxplot outliers pointtype 7')
gp.c('set style data boxplot')
gp.c('set boxwidth  0.5')
gp.c('set pointsize 0.5')

gp.c('unset key')
gp.c('set border 2')
gp.c('set xtics ("A" 1, "B" 2) scale 0.0')
gp.c('set xtics nomirror')
gp.c('set ytics nomirror')
gp.c('plot "plot_data.dat" using (1.0):1, "plot_data.dat" using (2.0):2')
gp.c('unset key')
gp.c('set grid')
gp.c('set xtics auto')
gp.c('plot "plot_data1.dat" lc rgb "red" lw 10 with dots , "plot_data2.dat" lc rgb "blue" lw 10 with dots')
gp.c('plot "plot_data1.dat" lc rgb "red" lw 10 with dots , "plot_data3.dat" lc rgb "green" lw 2 with lines')
gp.c('plot "plot_data2.dat" lc rgb "blue" lw 10 with dots , "plot_data3.dat" lc rgb "green" lw 2 with lines')
gp.c('plot "plot_data1.dat" lc rgb "red" lw 10 with dots , "plot_data4.dat" lc rgb "yellow" lw 2 with lines')
gp.c('plot "plot_data2.dat" lc rgb "blue" lw 10 with dots , "plot_data4.dat" lc rgb "yellow" lw 2 with lines')

rval_linear_train = r2_score(ytrain, lin)
rval_linear_test = r2_score(ytest, ypred)
rval_poly_train = r2_score(ytrain, p(xtrain))
rval_poly_test = r2_score(ytest, p(xtest))

print "The r score for linear train is {}".format(rval_linear_train)
print "The r score for linear test is {}".format(rval_linear_test)
print "The r score for poly train is {}".format(rval_poly_train)
print "The r score for poly test is {}".format(rval_poly_test)
