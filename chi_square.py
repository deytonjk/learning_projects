# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:28:28 2025

@author: josh_

AN EXERCISE IN CALCULATING CHI-SQUARE STATISTICS AND
USING THE scipy python library to do it

"""

import pandas as pd
import numpy as np



df = pd.read_csv(".\american_data.csv")
df1 = df # a spare copy
#TOTAL YES
total_yes = 0    
row_values = df.iloc[0].values  # or df.iloc[0].tolist()
for value in row_values[1:]:
# for funsies, i want to calculate the expected values using my own script
# let's get the row and column totals

    total_yes += value
    

#TOTAL NO
total_no = 0
row_values = df.iloc[1].values  # or df.iloc[0].tolist()
for value in row_values[1:]:
    total_no += value
    
    
# CALCULATE EXPECTED VALUES WITH FORMULA COL TOTAL X ROW TOTAL AND ADD TO 'EXPECTED' DATAFRAME
total_obs = total_yes + total_no

yes_row = ['Yes',]
no_row =['No']
title_row = []
for column in df.columns:
    title_row.append(column)
for column in df.columns[1:]:
    exp_yes = (total_yes*df[column].sum() / total_obs).astype('float64')
    exp_no = (total_no*df[column].sum() / total_obs).astype('float64')
    yes_row.append(exp_yes)
    no_row.append(exp_no)
    
combined = [title_row, yes_row, no_row]

expected_df = pd.DataFrame(combined, columns = combined[0])
expected_df=expected_df.drop(0)

# CHI SQUARE  = SUM((OBSERVED-EXPECTED)^2/EXPECTED)
# we don't need the happy column for this part
del expected_df['Happy']
del df['Happy']

expected_df.index = [0,1] # fixing the index to match df so the vectorization will work

chi_contribs = (df-expected_df)**2/expected_df  # here's the chi-square formula!

chi_square = chi_contribs.values.sum()

# Don't really have time to calculate the p-value with my own code...so use scipy
from scipy.stats import chi2_contingency

chi2_stat, p_value, dof, expected = chi2_contingency(df1)

print('The chi-square statistic is', chi_square, '.\nThe p-value here is', p_value,'.')

