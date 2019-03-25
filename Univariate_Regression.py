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
print u_quart
print l_quart
print u_quart + iqr
print l_quart - iqr
print good_val
l = int(0.8*df.shape[0])
print l
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
gp.s([x,y], filename='temp.dat')
gp.s([xtrain,ytrain], filename='temp1.dat')
gp.s([xtest,ytest], filename='temp2.dat')
gp.s([xtrain,lin], filename='temp3.dat')
gp.s([xp, p(xp)], filename='temp4.dat')

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
gp.c('plot "temp.dat" using (1.0):1, "temp.dat" using (2.0):2')
gp.c('unset key')
gp.c('set grid')
gp.c('set xtics auto')
gp.c('plot "temp1.dat" lc rgb "red" lw 10 with dots , "temp2.dat" lc rgb "blue" lw 10 with dots')
gp.c('plot "temp1.dat" lc rgb "red" lw 10 with dots , "temp3.dat" lc rgb "green" lw 2 with lines')
gp.c('plot "temp2.dat" lc rgb "blue" lw 10 with dots , "temp3.dat" lc rgb "green" lw 2 with lines')
gp.c('plot "temp1.dat" lc rgb "red" lw 10 with dots , "temp4.dat" lc rgb "yellow" lw 2 with lines')
gp.c('plot "temp2.dat" lc rgb "blue" lw 10 with dots , "temp4.dat" lc rgb "yellow" lw 2 with lines')

rval_linear = r2_score(ytest, ypred)
rval_poly_test = r2_score(ytest, p(xtest))
rval_poly_train = r2_score(ytrain, p(xtrain))

print "The r score for linear is {}".format(rval_linear)
print "The r score for poly train is {}".format(rval_poly_train)
print "The r score for poly test is {}".format(rval_poly_test)
