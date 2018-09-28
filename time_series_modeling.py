#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:41:36 2018

@author: jingjia
"""

import pandas as pd # not used
import numpy  as np 
import statsmodels.tsa.stattools as ts
from statsmodels.regression.linear_model import OLS, WLS, GLS
from datetime import datetime
import matplotlib.pyplot as plt


#------------Getting data-----------------------------------------------------
 
def loadDf1FromFile(fileName):
    return pd.read_csv(fileName,index_col='Date',na_values=['N/A'])
ImpVol = loadDf1FromFile('implied vol.csv')# implied vol 

#------------------Plot Input data --------------------------------------------

ImpVol.plot()
log_ImpVol = np.log(ImpVol)
log_ImpVol.plot()

#-------------------Get data list------------------------------------------
IV1M50D  = log_ImpVol['IV1M50d'].values.tolist()
Spot = log_ImpVol['Spot'].values.tolist()
RV1M = log_ImpVol['RV1M'].values.tolist()
Ratio_lag = log_ImpVol['Ratio.lag.RM'].values.tolist()

#------------Unit root testing-----------------------------------------------
res_IV1M50D=ts.adfuller(IV1M50D,regression="c",autolag=None,maxlag=1)
print ('IV1M50D ADF result:\n',res_IV1M50D[0:2])
print ('Test critical value\n',res_IV1M50D[4])
res_Spot=ts.adfuller(Spot,regression="c",autolag=None,maxlag=1)
print ('Spot ADF result:\n',res_Spot[0:2])
print ('Test critical value\n',res_Spot[4])
res_RV1M=ts.adfuller(RV1M,regression="c",autolag=None,maxlag=1)
print ('RV1M ADF result:\n', res_RV1M[0:2])
print ('Test critical value\n',res_RV1M[4])

y=np.array([x for x in Ratio_lag if x == x])


#--------------- OLS regression----------------------------------------------
num_vals1=len(y)
x1=np.array(IV1M50D[-num_vals1:]) 
x2=np.array(RV1M[-num_vals1:] ) 
x3=np.array(Spot[-num_vals1:] )
 
num_vals1=len(y)
b=np.vstack([x1,x2,x3,np.ones(num_vals1)]).T
test_beta1=OLS(y,b)


out=test_beta1.fit()
out.summary()
m1=out.params[0]
m2=out.params[1]
m3=out.params[2]

c=out.params[3]
print ('m1, m2, m3, c:     ', m1, m2, m3, c)
#import matplotlib.pyplot as plt
plt.plot(y, 'o', label='Original data', markersize=2)
plt.plot(m1*x1+m2*x2+m3*x3 + c, 'r', label='Fitted line')
plt.legend()
plt.show()


#--------------Regression residual unit testing------------------------------

residual=y-m1*x1-m2*x2-m3*x3-c
test_result=ts.adfuller(residual,regression="c",autolag=None,maxlag=1)
print ('Residual ADF result:\n', test_result[0:2])
print ('Test critical value\n',test_result[4])

test_result1=ts.adfuller(residual,regression="ct",autolag=None,maxlag=1)
print ('Residual ADF result:\n', test_result1[0:2])
print ('Test critical value\n',test_result1[4])

#The inclusion of a constant and trend in the test regression further shifts
# the distribution, result is good. the residual does not has an unit root
#problem and the residul is a stationary series at 1%,10% and 5% sifnificant 
#level)

plt.plot(residual)
plt.legend()
plt.show()
