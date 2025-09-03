# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:47:00 2025

@author: josh_
"""

import pandas as pd
from scipy import stats

# Load Ex 3 data file
emp_data = pd.read_csv("C://Users//josh_//Desktop//CSCI-5047 Data Analytics and Visualization//Salary Data - Ex3.csv")
print(emp_data.head())

#find the covariance
sal_ed_cov = emp_data['salary'].cov(emp_data['education'])
sal_pres_cov = emp_data['salary'].cov(emp_data['prestige'])
ed_pres_cov = emp_data['education'].cov(emp_data['prestige'])

#find the correlation coefficient and p-value
sal_ed_r, sal_ed_p = stats.pearsonr(emp_data['salary'], emp_data['education'])
sal_pres_r, sal_pres_p = stats.pearsonr(emp_data['salary'], emp_data['prestige'])
ed_pres_r, ed_pres_p = stats.pearsonr(emp_data['education'], emp_data['prestige'])


# display the results in a table
results_table = pd.DataFrame([[' ', 'Covariance', 'Correlation', 'p-Value'],
                              ['Salary/Education', sal_ed_cov, sal_ed_r, sal_ed_p],
                              ['Salary/Prestige', sal_pres_cov, sal_pres_r, sal_pres_p],
                              ['Education/Prestige', ed_pres_cov, ed_pres_r, ed_pres_p]
                              ])
# p-value was showing as 0.0 so this puts the table in scientific notation
pd.set_option('display.float_format', '{:.2E}'.format)

print(results_table)