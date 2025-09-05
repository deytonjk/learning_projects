# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 21:50:13 2025

@author: josh_
"""

import pandas as pd
from apyori import apriori

# load and check data
titanic_df = pd.read_csv("C:\\Users\\josh_\\Desktop\\CSCI-5047 Data Analytics and Visualization\\TitanicData.csv")

# in the excel file, the columns titles were misaligned, so i fixed that in excel before loading
# here i dropped the first column (now unnamed) because it was irrelevant to this process
titanic_df.drop(titanic_df.columns[0], axis=1, inplace=True)

# view the uniques in each column
for column in titanic_df.columns:
    print(f'Column uniques for {column}:', titanic_df[column].unique())

# get the rules and results
assoc_rules = apriori(titanic_df.values, min_support=0.005, min_confidence=0.8, min_length=2)
assoc_results = list(assoc_rules)

    
# filter the results, we just want survivors
filtered_results = []

for result in assoc_results:
    for entry in result.ordered_statistics:
        if entry.items_add == frozenset({'Yes'}):
            filtered_results.append(entry)
            
print('\nNumber of rules: ', len(filtered_results))

filtered_results.sort(key=lambda x: x.lift, reverse=True)

survived_df = pd.DataFrame(filtered_results)



# filter results for the passengers that did not survive (just curious)

filtered_results = []

for result in assoc_results:
    for entry in result.ordered_statistics:
        if entry.items_add == frozenset({'No'}):
            filtered_results.append(entry)
            
print('\nNumber of rules: ', len(filtered_results))

filtered_results.sort(key=lambda x: x.lift, reverse=True)

died_df = pd.DataFrame(filtered_results)



