data = read.csv("./income_dirty_data.csv", header = TRUE)
head(data)

# number of empty cells
empty_cells = sum(is.na(data))

# total number of cells (rows X columns)
cell_count = prod(dim(data))

# percent of cells complete
complete = (cell_count - empty_cells)/cell_count * 100

library(deducorrect)

cr = correctionRules(expression(if(!is.na(age) & age <18) age = NA, 
                                 if(!is.na(income) & income <1) income = NA, 
                                 if(!is.na(tax..15..) & tax..15..<1) tax..15.. = NA 
                                ))
corule = correctWithRules(cr, data)
newdata = corule$corrected


