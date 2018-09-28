#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 08:38:26 2018

@author: jingjia
"""

import pandas as pd
"""
Data load
"""
df = pd.DataFrame(pd.read_csv('PE_ratio.csv'))

"""
Clean the data and drop no PE ratio stocks from the dataframe
"""
df['PE_RATIO'] = pd.to_numeric(df['PE_RATIO'], errors='coerce')

df_nan = df[ df.isnull().any(axis=1)]

df = df.drop(df_nan.index, axis=0)
"""
Convert to percentile ranking within each sector. Then make it into Quntile
(It can make it to quintile directly. pct ranking is just more flexiable)
"""
df["rank"] = df.groupby(['INDUSTRY_SECTOR'])['PE_RATIO'].rank( pct=True)
df['Quintile']  = df['rank'].apply(lambda x : int(round(x*4,0)+1))

"""
Output file to verify
"""
writer = pd.ExcelWriter('Groupby_output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()