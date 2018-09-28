#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:10:32 2018

@author: jingjia
"""

import pandas as pd
import statsmodels.api as sm

"""
Data load
"""
df = pd.DataFrame(pd.read_csv('pairs-reg.csv'))


"""
Set up window length & calculate rolling beta
"""
window = 240
df['coeff']=None 
df['resid'] = None
df['z_score'] = None

df_pair = df.groupby(['tickerPair']).groups

for df_i in df_pair:
    pair_index= list(df_pair[df_i])
    print(pair_index[0],pair_index[-1])
    for i in range(pair_index[0]+window,pair_index[-1]):
        temp=df.iloc[i-window:i,:]
        RollOLS=sm.OLS(temp.loc[:,'Spread2.log'],temp.loc[:,['Spread1.log']]).fit()
        df.iloc[i,df.columns.get_loc('coeff')]=RollOLS.params[0]   

"""
The resid values = Y - X(given the PRIOR row's estimated beta)
"""
df['resid']=df['Spread2.log']- df['coeff'].shift(1)*df['Spread1.log']

"""
The Z_score need to calculate within each group
"""

for df_i in df_pair:
    pair_index= list(df_pair[df_i])
    for i in range(pair_index[0]+window+3,pair_index[-1]):
        temp=df.iloc[pair_index[0]+window+1:i]
        df.iloc[i,df.columns.get_loc('z_score')] = ( df.iloc[i,df.columns.get_loc('resid')] - temp['resid'].mean())/temp['resid'].std()
         
"""
Output file to verify
"""
writer = pd.ExcelWriter('rolling_reg_output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

