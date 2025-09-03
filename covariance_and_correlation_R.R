# Load Ex 3 data file
emp_data = read.csv("./salary_data3.csv", header = TRUE)
head(emp_data)

# pull out the desired columns
salary=emp_data$salary
education=emp_data$education
prestige=emp_data$prestige

# find the covariance
sal_ed_cov = cov(salary, education, method='pearson')
sal_pres_cov = cov(salary, prestige, method='pearson')
ed_pres_cov = cov(education, prestige, method='pearson')

# run the correlation test
sal_ed_comp = cor.test(salary, education, method='pearson')
sal_pres_comp = cor.test(salary, prestige, method='pearson')
ed_pres_comp = cor.test(education, prestige, method='pearson')

# pull the correlation coeff and p-values out of cor.test
sal_ed_r = sal_ed_comp$estimate
sal_ed_p = sal_ed_comp$p.value

sal_pres_r = sal_pres_comp$estimate
sal_pres_p = sal_pres_comp$p.value

ed_pres_r = ed_pres_comp$estimate
ed_pres_p = ed_pres_comp$p.value

# make a table to display the results

results= matrix(c(sal_ed_cov, sal_ed_r, sal_ed_p,
                  sal_pres_cov, sal_pres_r, sal_pres_p,
                  ed_pres_cov, ed_pres_r, ed_pres_p), ncol=3, byrow=TRUE)

# specify the column names and row names of matrix
colnames(results) = c('Covariance','Correlation','p-value')
rownames(results) <- c('Salary/Education','Salary/Prestige','Education/Prestige')

# assign a name to the table
results_output=as.table(results)

# display
results_output

