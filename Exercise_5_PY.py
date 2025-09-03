# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:31:32 2025

@author: josh_
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.compat import lzip
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

#load the data
album_df = pd.read_csv("C:\\Users\\josh_\\Desktop\\CSCI-5047 Data Analytics and Visualization\\Lab Album Sales.csv")

#see what it looks like
print(album_df.head())

###    SCATTERPLOTS  ####
plt.rcParams['figure.figsize'] = (10,8)
plt.title('Sales vs Ad Budget')
plt.scatter(x=album_df['AdvertBudget'], y=album_df['totalsales'], alpha=0.5, color='green')
plt.xlabel('Advertising Budget  ($ in thousands)')
plt.ylabel('Sales (thousands)')
plt.show()

plt.rcParams['figure.figsize'] = (10,8)
plt.title('Sales vs Airplay Times')
plt.scatter(x=album_df['AirplayTimes'], y=album_df['totalsales'], alpha=0.5, color='blue')
plt.xlabel('Airplay Times')
plt.ylabel('Sales (thousands)')
plt.show()

plt.rcParams['figure.figsize'] = (10,8)
plt.title('Sales vs Attractiveness Score')
plt.scatter(x=album_df['AttractivenessScore'], y=album_df['totalsales'], alpha=0.5, color='red')
plt.xlabel('Attractiveness')
plt.ylabel('Sales (thousands)')
plt.show()



#### Linear Regression Model (Single Variable)   ####
lm1 = smf.ols(formula='totalsales ~ AdvertBudget', data=album_df).fit()

print("\n                   LINEAR REGRESSION SUMMARY")
print(lm1.pvalues.to_string())
print(lm1.summary())
print('\nParameters')
print(lm1.params.to_string())

#prediction of sales with $135,000 advertising
pred_135 = int((lm1.params.iloc[1] * 135 + lm1.params.iloc[0])*1000)
print("\n\nPredicted album sales at $135k advertising budget:", pred_135)


# For fun: Scatterplot of adverts~sales with regression line
plt.rcParams['figure.figsize'] = (10,8)
plt.title('Sales vs Ad Budget')
plt.scatter(x=album_df['AdvertBudget'], y=album_df['totalsales'], alpha=0.5, color='green')
plt.xlabel('Advertising Budget ($ in thousands)')
plt.ylabel('Sales (thousands)')
    # add regression line
x = np.linspace(0, 2500, 10000) # Generates 400 points evenly spaced between -10 and 10
y = lm1.params.iloc[1] * x + lm1.params.iloc[0]
plt.plot(x, y, color='green')
plt.plot(135, 152.487, 'y*')
# Annotate the point with an arrow and text
plt.annotate(
    'Sales prediction with\n $135k in Ads: ~152k', 
    xy=(135, 152.487),  
    xytext=(1000, 100),  
    arrowprops=dict(arrowstyle='->', color='black'),  # Style of the arrow
    fontsize=10
)
plt.show()



#### Multiple Regression Model  ####

lm2 = smf.ols(formula='totalsales ~ AdvertBudget + AirplayTimes + AttractivenessScore', data=album_df).fit()
print("\n                   MULTIPLE REGRESSION SUMMARY")
print(lm2.pvalues.to_string())
print(lm2.summary())

#get a prediction using 135k sales budget + average airplay time and average attractiveness score
pred_dict={'AdvertBudget':135, 'AirplayTimes': np.mean(album_df['AirplayTimes']), 'AttractivenessScore': np.mean(album_df['AttractivenessScore'])}

print('\n\nModel 2 Prediction: ', lm2.predict(pred_dict), '\n\n')



