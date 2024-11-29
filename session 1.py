import numpy as np
from scipy import stats
from statistics import variance, stdev
import matplotlib.pyplot as plt


speed = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

#Calculates the mean of a data set, rounds it to the nearest whole number and prints
speed_mean = np.mean(speed) ; print ("Mean:", round(speed_mean,2))

#--^^-- Instead of mean, it does median
speed_median = np.median(speed) ; print ("Median:", speed_median)

#--^^-- Instead of median, it does mode
speed_mode = stats.mode(speed) ; print ("Mode:", speed_mode[0])



#75th Percentile is the point within a data set where 75% of the data is less than or equal to that point and 25% is more than that point
ages = np.array([5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31])

#Calculates the the 25th, 50th, 75th and 90th percentile of a data set
print ("Quantiles: ")
for val in [25, 50, 75, 90]:
    data_quantiles = np.percentile(ages,val)
    print (str(val) + "%", data_quantiles)


#Calculates the variance of a data set - 
    #Variance gives a sense of how far each data point is from the mean, on average.
    #High variance: The data points are widely spread out from the mean, indicating more variability.
    #Low variance: The data points are clustered closely around the mean, indicating less variability.
data_variance = variance(ages); print ("Sample Variance:", round (data_variance, 2))

#Calculates the Standard deviation -
    #If the standard deviation is small, the values are tightly clustered around the mean.
    #If the standard deviation is large, the values are more spread out.
data_Standard = stdev(ages); print ("Sample std.dev", round(data_Standard, 2))

#Calculates the range of a data set - the range is the difference between the lowest and highest point in the data set
data_range = np.max(ages, axis= 0) - np.min(ages, axis= 0); print ("Range:", data_range)



#Creates an array with 250 data points with values between 0 and 5
xtest = np.random.uniform(0.0, 5.0, 250)
#print(xtest)

#Create a bar graph visualising the array that was created, showcasing how many numbers are between 0 and 1, 1 and 2 etc
#plt.hist(xtest, 5)
#plt.show()

#A copy of the code above, how ever changing the number produced to be between 1 and 100, and creating 10000 data points
#ytest = np.random.uniform(0, 100, 10000)

#plt.hist(ytest, 100)
#plt.show()

#Creates a bell curve graph, commonly known as normal data distribution or the gaussian data distribution
#ztest = np.random.normal(5.0, 1.0, 100000)

#plt.hist(ztest, 100)
#plt.show()


#Creates a scatter plot diagram, using "atest" as the x axis and "btest" as the y axis
atest = [5,7,8,7,2,17,2,9,4,11,12,9,6]
btest = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(atest, btest)
plt.show()
#From the data you can identify anomallies, understand the relationship that a cars age has the the speed it can achieve and identify the highest data point and the lowest data point easily


#Creates a scatter plot diagram, using "ctest" as the x axis and "dtest" as the y axis, this plot will contain 1000 dots
ctest = np.random.normal(5.0, 1.0, 1000)
dtest = np.random.normal(10.0, 2.0, 1000)

plt.scatter(ctest, dtest)
plt.show()

#The dots concentrate between 12 and 8 on the y axis and 4 and 6 on the x axis
#The dots spread more on the y axis