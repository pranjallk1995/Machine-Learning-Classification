#K-NN Classifier

#inporting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = pd.DataFrame(dataset.iloc[:, [2, 3]])
Y = pd.DataFrame(dataset.iloc[:, 4])

#adding X_0
#X.insert(0, "X_0", 1)          #not required, handled by the package.

#performing feature scaling
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = X.astype('float')
X = pd.DataFrame(X_sc.fit_transform(X))

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

#fitting K-NN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier.fit(X_train, Y_train.values.ravel())

#visualising K-NN
Y_train = Y_train.values
for i in range(0, len(Y_train)):
    if Y_train[i] == 1:
        Y_train[i] = 25
    else:
        Y_train[i] = -25
class_no = np.ma.masked_where(Y_train > 0, Y_train)       # important class_yes and class_no appear to be interchanged.
class_yes = np.ma.masked_where(Y_train < 0, Y_train)
plt.scatter(X_train[0], X_train[1], marker = 'o', s = abs(class_yes), color = "blue")
plt.scatter(X_train[0], X_train[1], marker = 'x', s = abs(class_no), color = "red")
plt.title("Social Network training data")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

#predicting values
Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

#visualising K-NN on test data
Y_test = Y_test.values
for i in range(0, len(Y_test)):
    if Y_test[i] == 1:
        Y_test[i] = 25
    else:
        Y_test[i] = -25
class_no = np.ma.masked_where(Y_test > 0, Y_test)       # important class_yes and class_no appear to be interchanged.
class_yes = np.ma.masked_where(Y_test < 0, Y_test)
plt.scatter(X_test[0], X_test[1], marker = 'o', s = abs(class_yes), color = "blue")
plt.scatter(X_test[0], X_test[1], marker = 'x', s = abs(class_no), color = "red")
plt.title("Social Network test data")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

#performs slightly better than improved Logistic Regression on this dataset. (see confusion matrix).