import numpy as np
import pandas as pd

from pandas import Series, DataFrame

#Series_obj = Series(np.arange(10))
#print(Series_obj)


Series_obj = Series(np.arange(10), index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6','row 7', 'row 8', 'row 9', 'row 10'])
print(Series_obj)

print(Series_obj['row 7'])

print(Series_obj.iloc[[0, 7]])


#Generates random number, when using a "seed" it will always generate the same random numbers, this is to allow for consistency with random number generation for testing purposes
np.random.seed(1)

#Creates a data frame with 6 rows and 6 columns, and populates them with 36 ramonly generated numbers from seed 1
DF_obj = DataFrame(np.random.rand(36).reshape((6, 6)),
                   index=['row 1', 'row 2', 'row 3', 'row 4','row 5','row 6'],
                   columns=['column 1','column 2','column 3','column 4','column 5','column 6'])

#prints the Data frame
print(DF_obj)


#loc access rows and columns using actual labels rather than an integer position
print(DF_obj.loc[['row 2', 'row 5'], ['column 5', 'column 2']])

#prints the all data between and including row 2 and row 7
print(Series_obj['row 2': 'row 7'])


#Prints the data frame showing which data points are less than .2
print(DF_obj < .2)

