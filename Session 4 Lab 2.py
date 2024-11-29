from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

outlook=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

le = preprocessing.LabelEncoder()

outlook_encoded=le.fit_transform(outlook).tolist()
temp_encoded=le.fit_transform(temp).tolist()
label=le.fit_transform(play).tolist()

print('Outlook', outlook_encoded)
print ('Temp:',temp_encoded)
print ('Play:',label)

features=list(zip(outlook_encoded,temp_encoded))
print (features)

model = GaussianNB()

model.fit(features,label)

predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
if (predicted == 1):
   print ("Predicted Value:", 'Yes')
else:
   print ("Predicted Value:", 'No')

new_data = pd.read_csv('c:/Users/flof_/Documents/COM6033_Project_Assignment/AI Labs/Datasets/Lab_2_dataset.csv')

new_outlook_encoded = le.fit_transform(new_data['Outlook'])
new_temp_encoded = le.fit_transform(new_data['Temperature'])

new_features = list(zip(new_outlook_encoded, new_temp_encoded))

predicted_labels = model.predict(new_features)

for label in predicted_labels:
    if(label == 1):
      print("Yes")
    else:
      print("No")

expected_labels = le.fit_transform(new_data['Play Golf'])

accuracy = accuracy_score(expected_labels, predicted_labels)
print(f"Accuracy on new data: {accuracy * 100:.2f}%")