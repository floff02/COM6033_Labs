#imports
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

digits = load_digits()

#Prints the descriptions and characteristics of 
print(digits.DESCR)

#Prints the data between positions 1 and 20
print(digits.target[0:20])

#Displays the shape of the data set, with the result of (1797, 64), meaning there is 1797 rows with 64 columns
print(digits.data.shape)

#Prints the pixel values for 14th image in the dataset
print(digits.images[13])

#Prints what the intended number of the 14th image is
print(digits.target[13])

#Creates a grid of subplots with 4 rows and 6 columns and a figure size of 6" wide by 4" tall
figure, axes = plt.subplots(nrows = 4, ncols = 6, figsize = (6, 4))

#Shows the image plot
plt.show()

figure, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 3))

#Loop over each subplot axis, image and target label
for axes, image, target in zip(axes, digits.images, digits.target):

    #Turns off the axis lines and labels for cleaner display
    axes.set_axis_off()

    #Plots and displays the image onto the current axis in greyscale using nearest neighbour interpolation
    axes.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    #Sets the title of the subplot to the target number that is meant to be displayed within that subplot
    axes.set_title(target)

plt.show()

#Displays the 14th image from the dataset
image = plt.imshow(digits.images[13], cmap=plt.cm.gray_r)

plt.show()


#Splits the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(

    #Sets a random seed so that the split is reproducible
    #With 80% of the data being used for training and 20% being used for testing
    digits.data, digits.target, random_state=11, test_size=0.20)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

#Imports the KNeighborsClassifier from sklearn
knn = KNeighborsClassifier()

#Fit the KNN model using the training data
knn.fit(X=X_train, y=y_train)

#Using the model with the training data, predict the test data
predicted = knn.predict(X=X_test)

#Store the actual result for the test data
expected = y_test

#Print the first 20 predicted numbers
print(predicted[:20])

#Print the first 20 expected numbers
print(expected[:20])

#Create a list of tuples for incorrect predictions
wrong = [(int(p), int(e)) for (p, e) in zip(predicted, expected) if p != e]

#Print the list of wrong predictions 
print(wrong)

#Print the accuracy score of the modle on the test data as a percentage
print(f'{knn.score(X_test, y_test):.2%}')

#Generates a confusion matrix comparing predicted and expected
confusion = confusion_matrix(y_true=expected, y_pred=predicted)

#Prints the confusion matrix
print(confusion)

#Creates a list of digits for use in classification report
names = [str(digit) for digit in digits.target_names]

#Prints the classification report showing precision, recall, f1 score and support
print(classification_report(expected, predicted, target_names=names))

#Creates a data frame from confusion matrix for better visualisation
confusion_df = pd.DataFrame(confusion, index=range(10),
         columns=range(10))

#Creates heatmap to visualise the confusion matrix
axes = sns.heatmap(confusion_df, annot=True,
         cmap='nipy_spectral_r')
plt.show()

#Initialises KFold with 10 spilts, a random state for reproducibility
kfold = KFold(n_splits=10, random_state=11, shuffle=True)

# Performs cross-validation usin the KNN estimator on the whole digits data set, using the specified KFold
scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)

#Calculates the mean of the accuracy of the cross validation scores
print(f'Mean accuracy: {scores.mean():.2%}')

#Calculates the standard deviation of the cross validation scores
print(f'Accuracy standard deviation: {scores.std():.2%}')