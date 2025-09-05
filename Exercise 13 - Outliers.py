# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:16:49 2025

@author: josh_
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics as sts
import plotly.express as px

df = pd.read_csv("C:\\Users\\josh_\\Desktop\\Data Analytics and Visualization CSCI-5047\\Exercise 13 - Outliers\\carsDataset.csv")

cars_df = df[['cars.features', 'mpg', 'qsec', 'hp']]

# create boxplots of the data
fig, (ax1, ax2, ax3) = plt.subplots( 3, 1, figsize=(4,12))

sns.boxplot(data=cars_df['mpg'], ax=ax1)
ax1.set_title('Miles Per Gallon')

sns.boxplot(data=cars_df['qsec'], ax=ax2)
ax2.set_title('Quarter-Mile Split')

sns.boxplot(data=cars_df['hp'], ax=ax3)
ax3.set_title('Horsepower')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()



# outlier above or below 2 standard deviations from the mean
for column in cars_df.columns:
    if column != 'cars.features':
        avg = cars_df[column].mean()
        stdev = sts.stdev(cars_df[column])
        print(f"Outliers for {column}:")
        low = cars_df[cars_df[column]<(avg - 2*stdev) ] 
        high = cars_df[cars_df[column]>(avg+2*stdev)]
        outliers = pd.concat([low,high])
        print(outliers)
        print("\n\n")  



# make 3D plot
import plotly.io as pio

pio.renderers.default = 'browser'
        
fig = px.scatter_3d(cars_df, x = 'mpg', y = 'qsec', z = 'hp', text = 'cars.features', hover_data = 'cars.features')

fig.update_traces(textposition='top center')
        
fig.show()        





#### BANK DATA #####


bank_df = pd.read_csv("C:\\Users\\josh_\\Desktop\\Data Analytics and Visualization CSCI-5047\\Exercise 13 - Outliers\\bankloan.csv")

# create boxplots of the bank data
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots( 7, 1, figsize=(10,40))

sns.boxplot(data=bank_df['x1'], ax=ax1)
ax1.set_title('Age of Company (yrs)')

sns.boxplot(data=bank_df['x5'], ax=ax2)
ax2.set_title('Number of Managers')

sns.boxplot(data=bank_df['x6'], ax=ax3)
ax3.set_title('Avg Manager Age')

sns.boxplot(data=bank_df['x7'], ax=ax4)
ax4.set_title('Manager\'s Total Stock')

sns.boxplot(data=bank_df['x11'], ax=ax5)
ax5.set_title('Months Remaining on Obligation')

sns.boxplot(data=bank_df['x13'], ax=ax6)
ax6.set_title('Years Between Default')

sns.boxplot(data=bank_df['x14'], ax=ax7)
ax7.set_title('Years Between Last Payment')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()





# find top 10 multivariate outliers (Gower)

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

# scale the data
scaler = MinMaxScaler()
bankSeries = scaler.fit_transform(bank_df.fillna(0))

# cluster the data
outlier_detection = DBSCAN(min_samples=2, eps=1.13)
clusters = outlier_detection.fit_predict(bankSeries)

#trim dataframe to outliers from the cluster analysis
outliers = bank_df.iloc[(clusters == -1).nonzero()]

print(outliers)

outliers.to_csv('C:\\Users\\josh_\\Desktop\\Data Analytics and Visualization CSCI-5047\\Exercise 13 - Outliers\\bank_outliers.csv')


