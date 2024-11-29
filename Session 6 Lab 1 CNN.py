#Import the tensorflow.keras.datasets.mnist module so we can load the dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from IPython.display import Image
#Import matplotlib
import matplotlib.pyplot as plt
#Import seaborn
import seaborn as sns

#Now load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)   
print(y_train.shape)  
print(X_test.shape)   
print(y_test.shape)

#Select the font scale
sns.set(font_scale=2)

#Run this snippet multiple times to see additional randomly selected digits.
import numpy as np
index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))

for item in zip(axes.ravel(), X_train[index], y_train[index]):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([]) # remove x-axis tick marks
    axes.set_yticks([]) # remove y-axis tick marks
    axes.set_title(target)
plt.tight_layout()
plt.show()

#CNN require images to be in the shape (width, height, channels)
#Reshape the X_train dataset
#Why channels is 1 in this case, answer using another # line
#The 1 channel is the set the colour scheme of the images to greyscale
X_train = X_train.reshape((60000, 28, 28, 1))

#Check the shape now, what has changed
print(X_train.shape)

#Now do the same for X_test
X_test = X_test.reshape((10000, 28, 28, 1))
    
#Check the shape now
print(X_test.shape)

#Do you still remember what is normalisation?
#Normalisation is the practice of dividing the values of a dataset so that the values are between 0 and 1 to ensure features with larger ranges dont dominate the model
#Now normalise the X_train
X_train = X_train.astype('float32') / 255
     

#Do the same for X_test
X_test = X_test.astype('float32') / 255

y_train = to_categorical(y_train)
print(y_train.shape)
     
y_train[0]
     
y_test = to_categorical(y_test)
print(y_test.shape)
#Explain what has changed and why using a new # line
#The shape of y_train and y_test have changed from (60000,) and (10000,) to (60000, 10) and (10000, 10) this is due to the labels within the dataset being converted from integers to messurable classes which can be used for validation and accuracy tracking

cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#Define activation function?
#Activation functions determine if a particular neuron should be activated and passed to the next layer

cnn.add(MaxPooling2D(pool_size=(2, 2)))
#What is max pooling techanique and why it is used?
#Down sizes the image by taking 2x2 areas of the images and and taking the maximun value within and applying it the the other 3. Essentially making the image 14x14 and reducing computational power needed and controlling overfitting

cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=10, activation='softmax'))

cnn.summary()

#plot_model(cnn, to_file='convnet.png', show_shapes=True,
#            show_layer_names=True)
#Image(filename='convnet.png')

cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=5, batch_size=64,
              validation_split=0.1)

loss, accuracy = cnn.evaluate(X_test, y_test)

print(loss)
print(accuracy)


predictions = cnn.predict(X_test)
     
y_test[0]
     

for index, probability in enumerate(predictions[0]):
          print(f'{index}: {probability:.10%}')

images = X_test.reshape((10000, 28, 28))
incorrect_predictions = []

for i, (p, e) in enumerate(zip(predictions, y_test)):
    predicted, expected = np.argmax(p), np.argmax(e)

    if predicted != expected:
        incorrect_predictions.append(
            (i, images[i], predicted, expected))
     

len(incorrect_predictions)


figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 12))

for axes, item in zip(axes.ravel(), incorrect_predictions):
    index, image, predicted, expected = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([]) # remove x-axis tick marks
    axes.set_yticks([]) # remove y-axis tick marks
    axes.set_title(
        f'index: {index}\np: {predicted}; e: {expected}')
plt.tight_layout()
plt.show()

cnn.save('Session_6_Lab_1_CNN.h5')

cnn = load_model('Session_6_Lab_1_CNN.h5')

cnn.summary()