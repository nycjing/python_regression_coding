#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 23:38:52 2018

@author: jingjia
"""


import pandas as pd
"""
load data to dataframe
"""
df = pd.DataFrame(pd.read_csv('Python munging.csv'))

"""
create one more column that join CMATICKER and TENOR with '-'
"""
df['CMATICKER-TENOR'] = df['CMATicker'] + '-' + df['Tenor'].map(str) 

"""
Fill missing ClientEntityID with CMATICKER-TENOR and set it as index
"""
df['ClientEntityId']= (df['ClientEntityId']).fillna(df['CMATICKER-TENOR'])
df.set_index(['ClientEntityId'],inplace=True)
