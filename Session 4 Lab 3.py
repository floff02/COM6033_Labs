import graphlib
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz


df = pd.read_csv("c:/Users/flof_/Documents/COM6033_Project_Assignment/AI Labs/Datasets/Lab_3_dataset.csv")

print(df)
#The read_csv function will simply read and extract information from CSV files allowing for the data to be visualised and manipulated
#The label from the dataset imported above is GO, labeling wethere the person went to the comedy show or not

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)

e = {"YES" : 1, "NO": 0}
df["Go"] = df['Go'].map(e)
#Map changes any data with the given value into the specified value set, in this case any data with the value of "YES" in the column of "Go" is changed into a "1"


le = preprocessing.LabelEncoder()

df['Go'] = le.fit_transform(df['Go'])

print(df)


# Features list (titles should be the same as in the dataset)
features = ['Age', 'Experience', 'Rank', 'Nationality']
# Split the features from their labels
X = df[features]
y = df['Go']

print(X.shape)

print(y.shape)

print(X.describe())

print(y.describe())

clf = LogisticRegression()
clf = clf.fit(X.values, y)

prediction = clf.predict([[40, 10, 7, 1]])

if prediction == 1:
    print("Given the values inputed, you should go")
else:
    print("Given the values inputed, you shouldn't go")

print(clf.predict_proba([[40, 10, 7, 1]]))
#This prints to probability of wether you should or souldnt go. [[0.41174268 0.58825732]] this means that there is a 41.17% chance that you shouldnt go, and a 58.82% chance that you should go

input_data = pd.DataFrame([[40, 10, 7, 1]], columns = features)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

prediction = clf.predict(input_data)

if prediction == 1:
    print("Given the values inputed, you should go")
else:
    print("Given the values inputed, you shouldn't go")


print(clf.predict_proba([[40, 10, 7, 1]]))


dot_data = tree.export_graphviz(
    clf, 
    out_file=None,  # Do not write to a file
    feature_names=features,  # Use feature names
    class_names=['No', 'Yes'],  # Class labels
    filled=True,  # Fill nodes with colors
    rounded=True,  # Use rounded nodes
    special_characters=True  # Handle special characters
)

graph = graphviz.Source(dot_data)

graph.view()