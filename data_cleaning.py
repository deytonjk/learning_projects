# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:57:11 2025

@author: josh_
"""

import pandas as pd
import random
import numpy as np

# Load Income Dirty Data file
people_df_dirty = pd.read_csv("./income_dirty_data.csv", index_col='ID')

print(people_df_dirty.head())
print(people_df_dirty.shape)
      
# Find number of empty or missing cells
missing_cells = people_df_dirty.isnull().sum().sum()  
percent_complete_cells = 100 - round(missing_cells/people_df_dirty.size*100,2)
print('\nPercentage of non-empty cells in the dataframe:', percent_complete_cells,'%')        

# Find number and percent of incomplete observations (rows with at least 1 missing value)
rows_missing_1 = people_df_dirty.isnull().any(axis=1).sum()
    
percent_complete_rows = 100 - rows_missing_1/people_df_dirty.shape[0]*100   
print('\nPercentage of complete rows:', percent_complete_rows,'%')

# Find the number of rows with no errors in age, income, and tax
no_a_i_t_errors =len(
people_df_dirty[
    (   # all are 18 and older 
        people_df_dirty['age'] >= 18
    ) 
    &
    (   # tax is 15% of income - exactly 15% would be affected by rounding errors so i added a range instead of straight ==
        (people_df_dirty['tax (15%)']/people_df_dirty['income']>0.14) & (people_df_dirty['tax (15%)']/people_df_dirty['income']<0.16)
    )
    & 
    (   # emps shouldn't be working for free
        people_df_dirty['income']>0 
    ) 
    ]

)

# Print the percentage of rows with no age, income, or tax errors
print('Percentage of rows with no age, income, or tax errors:', no_a_i_t_errors/len(people_df_dirty)*100,'%')

##### CLEANING UP THE GENDER COLUMN  ####

# We want to edit a copy of the original data frame
fix_gender_df = people_df_dirty

# Change all non-empty cells to Male or Female (consistency)

fix_gender_df['gender'] = fix_gender_df['gender'].str.replace('Man', 'Male', regex=True)
fix_gender_df['gender'] = fix_gender_df['gender'].str.replace('Men', 'Male', regex=True)
fix_gender_df['gender'] = fix_gender_df['gender'].str.replace('Women', 'Female', regex=True)
fix_gender_df['gender'] = fix_gender_df['gender'].str.replace('Woman', 'Female', regex=True)

# check to make sure we've changed all non-empty cells
print(fix_gender_df['gender'].unique())

# To fill in the empty gender cells, get a ratio of male to female in the known set
m2f_ratio = (len(fix_gender_df[
       # male 
        fix_gender_df['gender'] == 'Male'
    ])) / (len(fix_gender_df[
       # female 
        fix_gender_df['gender'] == 'Female'
    ]))

#convert to a percent for the random choice method
male_percent = m2f_ratio/(1+m2f_ratio)*100
female_percent = 100-male_percent

# fill in the empty cells with random Male/Female choice weighted by the ratio
choices = ['Male', 'Female']
fix_gender_df['gender'] = fix_gender_df['gender'].fillna(random.choices(choices, weights = [male_percent, female_percent], k=1)[0])

# check the uniques in gender again
print(fix_gender_df['gender'].unique())


### CLEANING UP THE INCOME COLUMN ###

# fix the incomes that have valid tax value observations
fix_gender_df['income'] = fix_gender_df.apply(lambda row: row['tax (15%)'] / 0.15 if ((row['income']<=0 or pd.isna(row['income'])) & (row['tax (15%)']>0)) else row['income'], axis=1)
        
# i noticed some of the income values were less than the tax values (e.g. one income was just 70)
# replacing these incomes using the apply() feature as before
fix_gender_df['income'] = fix_gender_df.apply(lambda row: row['tax (15%)'] / 0.15 if (row['income']>0) & (row['income']<row['tax (15%)']) else row['income'], axis=1)

# some incomes aren't negatives or missing but still do not make sense, if they are just two digits, i assume they meant something like 70k so I added 3 zeros
# if they are 3 or 4 digits, that's likely a monthly salary, so I'm multiplying those by 12

fix_gender_df['income'] = fix_gender_df.apply(lambda row: row['income'] * 1000 if ((row['income']>0) & (row['income']<=150)) else row['income'], axis=1)
fix_gender_df['income'] = fix_gender_df.apply(lambda row: row['income'] * 12 if ((row['income']>700) & (row['income']<10000)) else row['income'], axis=1)



# replace the rest of the income values with NaN
fix_gender_df['income'] = fix_gender_df.apply(lambda row: np.NAN if (row['income']<=0)  else row['income'], axis=1)


# CHECK AND FIX THE TAX VALUES

fix_gender_df['tax (15%)'] = fix_gender_df.apply(lambda row: round(row['income'] * 0.15) if (row['income']>0) else row['tax (15%)'], axis=1)


#fill the rest of the tax values with NaN

fix_gender_df['tax (15%)'] = fix_gender_df.apply(lambda row: np.NaN if (row['tax (15%)']<=0) else row['tax (15%)'], axis=1)


# just applying a more appropriate name to the data frame
tig_cleaned_df = fix_gender_df

# NOT IN THE INSTRUCTIONS, BUT CHANGING INVALID AGE VALUES TO NAN
tig_cleaned_df['age'] = tig_cleaned_df.apply(lambda row: np.NaN if (row['age']<18) else row['age'], axis=1)



# VIEW SUMMARY
# 
pd.options.display.float_format = '{:.2f}'.format

print(tig_cleaned_df.describe())

cols=['gender','age', 'income', 'tax (15%)']
stat_list = [['Column', 'Number of NaN']]
for column in cols:
    NumOnan = tig_cleaned_df[column].isna().sum()
    stat_list.append([column, NumOnan])
print(pd.DataFrame(stat_list))


# Transform Gender column from Male, Femal to 0,1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
tig_cleaned_df['gender'] = le.fit_transform(tig_cleaned_df['gender'])

# Rescale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = [['gender', 'age', 'income', 'tax (15%)']]
for feature in features:
    tig_cleaned_df[feature]=scaler.fit_transform(tig_cleaned_df[feature])


# Impute
from sklearn.impute import KNNImputer

imputer = KNNImputer()

m_filled_data = imputer.fit_transform(tig_cleaned_df)

final_people_df=pd.DataFrame(data=m_filled_data, columns=features, index=range(1,1001))
final_people_df.index.name="ID"



print('\n\nSummary after KNN Imputation')
print(final_people_df.describe())
stat_list = [['Column', 'Number of NaN']]
for column in cols:
    NumOnan = final_people_df[column].isna().sum().iloc[0]
    stat_list.append([column, NumOnan])
print(pd.DataFrame(stat_list))


