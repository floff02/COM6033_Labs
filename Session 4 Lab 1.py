import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

nyc = pd.read_csv('c:/Users/flof_/Documents/COM6033_Project_Assignment/AI Labs/Datasets/Lab_1_dataset.csv')

nyc.columns = ['Date', 'Temperature', 'Anomaly']

#print(nyc.head(3))

nyc.Date = nyc.Date.floordiv(100)

print(nyc.head(3))

print(nyc.tail(3))

print(nyc.describe())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
     nyc.Date.values.reshape(-1, 1), nyc.Temperature.values,
     random_state=11)

print(X_train.shape)

print(X_test.shape)

print(X_train)

print(y_train)


linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)

print(linear_regression.intercept_)

predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
     print(f'predicted: {p:.2f}, expected: {e:.2f}')


predict = (lambda x: linear_regression.coef_ * x +
                      linear_regression.intercept_)

print(predict(2019))

print(predict(1890))


axes = sns.scatterplot(data=nyc, x='Date', y='Temperature',
     hue='Temperature', palette='winter', legend=False)

axes.set_ylim(10, 70)

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

y = predict(x)

line = plt.plot(x, y)

plt.show()

print(predict(2020))

print(predict(1889))
