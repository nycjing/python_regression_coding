#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:12:17 2018

@author: jingjia
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

"""
Data load, using train_test_split create train_set & test_set
"""

train = pd.read_csv('Boston.csv')
included_features = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black',
                    'lstat'] 
# define the training data X...
X = train[included_features]
Y = train[['medv']]



"""
using train_test_split create train_set & test_set
"""
X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.5, random_state=0)


"""
Linear Regression using train set & plot predict with test set 
"""
lm = LinearRegression()
lm.fit(X_train, y_train)

pred_lm = lm.predict(X_test)
mse = mean_squared_error(y_test,pred_lm)

print('Linear Regression MSE:',mse)


"""
Decision Tree
"""
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import cross_val_score
# define the training data X...

#X_train = X_train[['ptratio','dis','age','rm']]
#Y_train = y_train[['medv']]
##
#X_test = X_test[['ptratio','dis','age','rm']]
#Y_test = y_test[['medv']]

# try fitting a decision tree regression model...
DTR_1 = dtr(max_depth=None) # declare the regression model form. Let the depth be default.
DTR_1.fit(X_train, y_train) # fit the training data
pred_DTR = DTR_1.predict(X_test)

mse = mean_squared_error(y_test,pred_DTR)

print('Decision Tree regression MSE:',mse)

#plt.plot(y_test, 'co', label='True data')
#plt.plot(pred_DTR, 'mo', label='Decision Tree predict')
#plt.legend(loc='best');

#scores_dtr = cross_val_score(DTR_1, X, Y, cv=10,scoring='explained_variance') # 10-fold cross validation
#print('scores for k=10 fold validation:',scores_dtr)
#print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_dtr.mean(), scores_dtr.std() * 2))
"""
Random Forest
"""

from sklearn.ensemble import RandomForestRegressor as rfr

estimators = [10, 25, 50, 75]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
yt = [i for i in Y['medv']] 
np.random.seed(0)
for i in estimators:
    model = rfr(n_estimators=i,max_depth=None)
    model.fit(X,yt)
    mse = mean_squared_error(y_test,model.predict(X_test))
    print('Random Forest regression MSE:',mse)
    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
    print('estimators:',i)
    print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
    print('')
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting

"""
Feature selection
"""    
import sklearn.feature_selection as fs
mir_result = fs.mutual_info_regression(X, yt) # mutual information regression feature ordering
feature_scores = []
for i in np.arange(len(included_features)):
    feature_scores.append([included_features[i],mir_result[i]])
sorted_scores = sorted(np.array(feature_scores), key=lambda s: float(s[1]), reverse=True) 
print(np.array(sorted_scores))
    