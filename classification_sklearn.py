# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:40:48 2025

@author: josh_
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv(".\classification_lab_data.csv")

print(data.head())


# checkout the columns
print(data.dtypes)

# for the non numerics, lets check out the uniques
for column in data.columns:
    if data[column].dtype=='object':
        print(column)
        print(data[column].unique())
        
        
# check for missing data
print(data.isnull().sum())
print(data.isna().sum())


inputs = data.drop('Label', axis=1)
target = data['Label']

# preprocessing - encode the object cols as numbers
worktype_code = LabelEncoder()
EducationLevel_code = LabelEncoder()
marital_status_code = LabelEncoder()
CurrentOccupation_code = LabelEncoder()
RelationshipStatus_code = LabelEncoder()
race_code = LabelEncoder()
Gender_code = LabelEncoder()
Label_code = LabelEncoder()

inputs['worktype_n']=worktype_code.fit_transform(inputs['worktype'])
inputs['EdLevel_n']=EducationLevel_code.fit_transform(inputs['EducationLevel'])
inputs['Married_n']=marital_status_code.fit_transform(inputs['marital_status'])
inputs['Occupation_n']=CurrentOccupation_code.fit_transform(inputs['CurrentOccupation'])
inputs['Relationship_n']=RelationshipStatus_code.fit_transform(inputs['RelationshipStatus'])
inputs['Race_n']=race_code.fit_transform(inputs['race'])
inputs['Gender_n']=Gender_code.fit_transform(inputs['Gender'])


columns_to_drop = ['worktype', 'EducationLevel', 'marital_status', 'CurrentOccupation', 'RelationshipStatus', 'race', 'Gender']
inputs_n = inputs.drop(columns_to_drop, axis=1)    


x = inputs_n
y = target

# generate some training and testing data
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=.20, random_state=42, stratify=y)


# RUN DECISION TREE MODEL
clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

results = metrics.confusion_matrix(y_test, y_pred)

print('\n\nDECISION TREE')
print('\nConfusion Matrix\n',results)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

print('tn, fp, fn, tp', tn, fp, fn, tp)

accuracy = (tp + tn) / (tn + fp + fn + tp)

print('Decision Tree Accuracy: ', accuracy)

sensitivity = tp/(tp + fn)

specificity = tn / (tn + fp)

print( 'Decision Tree Sensitivity: ', sensitivity)
print('Decision Tree Specificity:', specificity)



# run the Bayes classification
NBmodel = GaussianNB()

NBmodel.fit(X_train, y_train)

y_pred = NBmodel.predict(X_test)

results = metrics.confusion_matrix(y_test, y_pred)

print('\n\nNAIVE BAYES GAUSSIAN')
print('\nConfusion Matrix\n',results)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

print('tn, fp, fn, tp', tn, fp, fn, tp)

accuracy = (tp + tn) / (tn + fp + fn + tp)

print('Naive Bayes Accuracy: ', accuracy)

sensitivity = tp/(tp + fn)

specificity = tn / (tn + fp)

print( 'Naive Bayes Sensitivity: ', sensitivity)
print('Naive Bayes Specificity:', specificity)


