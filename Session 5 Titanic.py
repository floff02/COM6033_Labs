from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import pickle
import joblib

df = pd.read_csv('c:/Users/flof_/Documents/COM6033_Project_Assignment/AI Labs/Datasets/Lab_Titanic_dataset.csv')
orig_df = df

#Explore the Titanic dataset
#Print all samples and check how many samples and features the Titanic dataset has
#The titanic dataset has 1309 samples
print(orig_df)

#Check the type of each variable (data type)
#int type is ok, float64 is ok as well but you may change it to int, object type need to be changed to int (object is a string in pandas and perform a string operation)
print(df.dtypes)

#Use shape attribute to check the raws (samples) and columns (features)
print(df.shape)

#Use describe attribute to explore the data statistics
#Can you let me a little bit about the data, for example the age groups
#The ages range from 0.17 to 80 with a mean of 29.88 and a count of 1046
print(df.describe())

#Use describe attribute at different location to explore the data statistics
#Use 3 or 4 instead of 2 to include more features (this is useful when you have lots of features)
df.describe().iloc[:,:3]

#Use isnull() to find columns or rows with missing values and sum them up to get the total of missing values
#Which features are the leak features?
#Cabin, Boat, Body and home.dest
df.isnull().sum()

#We can create a boolean array (a series with True or False to indicate if a row (a sample) has missing data)
#and use it to inspect rows that are missing data
mask = df.isnull()

mask.head()  # rows

#Let's improve the process by using the function any that iterate through each row and return true for any x in the raw = true
mask = df.isnull().any(axis=1)
mask.head()
df[mask].body.head() # check body column
df[mask].age.head() # check age column
df[mask].embarked.head() # check embarked column

#Use the .value_counts method to examine the counts of the values:
df.sex.value_counts(dropna=False) # How many male and female
# Assign dropna to false if you don't want to delete the missing values

#Use the .value_counts method to examine the counts of the values:
df.embarked.value_counts(dropna=False)

#Use the .value_counts method to examine the counts of the values:
df.age.value_counts(dropna=False)

#Delete raws with high percentage of missing values
df = df.drop(
     columns=[
         "name",
         "ticket",
         "home.dest",
         "boat",
         "body",
         "cabin",
     ]
 )

#Use the attribute describe to check whether you managed to delete the columns
#Compare it with the above df.describe()
print(df.describe())

print(df.shape)






#Populate age missing values with thier median

df['age'] = df['age'].fillna(df['age'].median())

#Populate embarked missing values with high occurrence value

df['embarked'] = df['embarked'].fillna('S')

# map sex to a numeric type
df.sex = df.sex.map({'male': 1, 'female': 0})

# map embarked to a numeric type
df.embarked = df.embarked.map({'S': 2, 'C': 1, 'Q':0})

#fill any other missing value with 0 (is not good practice but to avoid common error of NaN value still exist)
df.fillna(0,inplace=True)

#print(df.sex)




#Assign survived column (targets) to y
y = df.survived
#Delete survived column from X (samples)
X = df.drop(columns="survived")
#Now split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
#check the y_train (target)
#print(y_train)
#check the X_train (samples)
#print(X_train)


clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
#Get the predicted and expected
#Can you tell what is predicted and expected values represent?
#Can you derive the misclassified values (wrong)
predicted = clf.predict(X=X_test)
expected = y_test
#Now print the model accuracy
print(f'{clf.score(X_test, y_test):.2%}')
clf.predict(X_test)




kfold = KFold(n_splits=10, random_state=11, shuffle=True)

scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print(f'Mean accuracy: {scores.mean():.2%}')
#Calculates the standard deviation of the cross validation scores
print(f'Accuracy standard deviation: {scores.std():.2%}')

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

confusion_df = pd.DataFrame(confusion, index=range(2),
         columns=range(2))

axes = sns.heatmap(confusion_df, annot=True,
         cmap='nipy_spectral_r')
plt.show()


print(df.shape)

saved_model = pickle.dumps(clf)

# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions
print(clf_from_pickle.predict(X_test))

joblib.dump(clf, 'c:/Users/flof_/Documents/COM6033_Project_Assignment/Titanic_model.pkl')

clf_from_joblib = joblib.load('c:/Users/flof_/Documents/COM6033_Project_Assignment/Titanic_model.pkl')

# Use the loaded model to make predictions
print(clf_from_joblib.predict(X_test))