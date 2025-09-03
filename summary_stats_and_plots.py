# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:16:34 2025

@author: josh_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


#Load the data
salary_df = pd.read_csv(".\salary_data2.csv")
print(salary_df.head())

#Calculate the stats for the requested columns and put in a easily viewable table
stat_list = [['Column', 'Min Value', 'Q1', 'Median', 'Mean', 'Q3', 'Max Value']]
for column in salary_df.columns:
    if salary_df[column].dtype == 'int64':
        minv = np.min(salary_df[column])
        q1 = np.percentile(salary_df[column], 25)
        q2 = np.median(salary_df[column])
        pie = round(np.mean(salary_df[column]),2)
        q3 = np.percentile(salary_df[column], 75)
        maxv = np.max(salary_df[column])
        stat_list.append([column, minv, q1, q2, pie, q3, maxv])
print(stat_list)        

stats_df = pd.DataFrame(stat_list)
print(stats_df)

# Prestige DISTRIBUTION
n, bins, patches = plt.hist(salary_df['prestige'], bins=10, color='g', alpha=0.7, edgecolor='black', linewidth=1 )
plt.title('Prestige Distribution', color = 'g')
plt.xlabel('Prestige', color = 'g')
plt.ylabel('Frequency', color = 'g')

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.show()

# Education DISTRIBUTION
n, bins, patches = plt.hist(salary_df['education'], bins=10, color='b', alpha=0.7, edgecolor='black', linewidth=1 )
plt.title('Education Distribution', color = 'b')
plt.xlabel('Education', color = 'b')
plt.ylabel('Frequency', color = 'b')

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.show()

# Scatter plots of prestige vs education vs salary
plt.rcParams['figure.figsize'] = (8.9, 5) #sets the size of the scatterplots - I chose a golden rectangle for the length/width ratios

plt.suptitle('Salary, Prestige, and Education')
plt.subplot(1,2,1)
plt.scatter(x=salary_df['education'], y=salary_df['salary'], alpha=0.5, color='red')
plt.xlabel('Education')
plt.ylabel('Salary ($)')

plt.subplot(1,2,2)
plt.scatter(x=salary_df['education'], y=salary_df['prestige'], alpha=0.5, color='blue')
plt.xlabel('Edcuation')
plt.ylabel('Prestige')
plt.show()

#Find the variance and standard deviation of the salary column

salary_var = round(stats.variance(salary_df['salary']), 2)
salary_sdv = round(stats.stdev(salary_df['salary']), 2) # or we could have just taken the square root of the variance...

print('The variance of salaries in this table is', salary_var, '.')
print('The standard deviation of salaries in this table is', salary_sdv, '.')

