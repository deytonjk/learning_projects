# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:33:07 2025

USING scipy AND scikit-learn TO GET STATISTICS ON NETWORK TRAFFIC DATA
AND CLASSIFY TRAFFIC AS NORMAL OR SUSPICIOUS


@author: josh_
"""

import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load data
net_data = pd.read_csv(".\network_traffic_data.csv")
print(net_data.head())

for column in net_data.columns:
    if net_data[column].var() > 5:
        print (f'{column} variance:' ,net_data[column].var())

for column in net_data.columns:
    if column != 'Label':
        pearson_corr_sp, p_value = pearsonr(net_data[column], net_data['Label'])
        if pearson_corr_sp > 0.5:
            print(column)
            print(f"Pearson correlation: {round(pearson_corr_sp, 2)}, p-value: {p_value}")


# separate the attributes from the variable we're trying to predict
net_attribs = net_data.drop('Label', axis='columns')

x = net_attribs
y = net_data['Label']

# generate some training and testing data
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=.33, random_state=42)

# run the Bayes classification
model1 = GaussianNB()

model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

# results  - i added in a more colorful confusion matrix in the plots
print('\n \n')
print("Naive Bayes")


cm1 = confusion_matrix(y_test, y_pred1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
print('Confusion Matrix')
print(cm1)               
print(metrics.classification_report(y_test, y_pred1))               


disp.plot()

plt.show()










