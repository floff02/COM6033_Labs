from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

digits = load_digits()

for i in range(0, len(digits.target), 50):
    print(f'Index {i}: Label {digits.target[i]}')

print(digits.data.shape)

image = plt.imshow(digits.images[20], cmap=plt.cm.gray_r)

plt.show()

print(digits.target[20])

figure, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 3))


for axes, image, target in zip(axes, digits.images, digits.target):

    axes.set_axis_off()

    axes.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    axes.set_title(target)

plt.show()


X_train, X_test, y_train, y_test = train_test_split(
digits.data, digits.target, random_state=11, test_size=0.20)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

svc = SVC()

svc.fit(X=X_train, y=y_train)

predicted = svc.predict(X=X_test)

expected = y_test

print(predicted[:20])

print(expected[:20])

wrong = [(int(p), int(e)) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)

print(f'{svc.score(X_test, y_test):.2%}')

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

names = [str(digit) for digit in digits.target_names]

print(classification_report(expected, predicted, target_names=names))
#Has the same percentage of accuracy, how ever it seems to be more consitantly miss predicting the number 3