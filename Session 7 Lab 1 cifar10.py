from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(font_scale=2)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#The cifar10 dataset has 60000 samples

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

#Now normalise the X_train
X_train = X_train.astype('float32') / 255
#Do the same for X_test
X_test = X_test.astype('float32') / 255

num_classes = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

y_train = to_categorical(y_train, num_classes)
print(y_train.shape)
y_train[0]
y_test = to_categorical(y_test, num_classes)
print(y_test.shape)


cnn = Sequential()

cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(units=10, activation='softmax'))

cnn.summary()

#plot_model(cnn, to_file='convnet.png', show_shapes=True,
#            show_layer_names=True)
#Image(filename='convnet.png')

cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=20, batch_size=16,validation_split=0.01)

loss, accuracy = cnn.evaluate(X_test, y_test)

print(loss)
print(accuracy)

#First run with code from Labs gave an accuracy score of 67.61%
#Second run with changes made the the convolution layers, changing the kernel size to (3,3) and adding dropout layers increased the accuracy to 71.31%
#Third run with changes to the model adding batch normalisation, more dropout layers and decreasing the batch size from 32 to 16 this increased the accuracy to 76.69%

predictions = cnn.predict(X_test)
y_test[0]
for index, probability in enumerate(predictions[0]):
          print(f'{index}: {probability:.10%}')

images = X_test
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

def display_probabilities(prediction):
    for index, probability in enumerate(prediction):
        print(f'{index}: {probability:.10%}')
display_probabilities(predictions[495])
display_probabilities(predictions[583])

cnn.save('Session_7_Lab_1_cifar10_cnn.h5')

cnn = load_model('Session_7_Lab_1_cifar10_cnn.h5')

cnn.summary()