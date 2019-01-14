#Linear Logistic Regresssion

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

#adding polynomial terms
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 2)
X = pd.DataFrame(poly_regressor.fit_transform(X))

#performing feature scaling
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = X.astype('float')
X = pd.DataFrame(X_sc.fit_transform(X))

#print(X)

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

#fitting the Logistic Regression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0, solver = 'lbfgs')
regressor.fit(X_train, Y_train.values.ravel())

#predicting values
Y_pred = regressor.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

"""
Confusion Matrix without polynomial features.
[[82  7]  [12 39]]

Confusion Matrix with ploynomial features.
[[83  6]  [10 41]]

Therefore, less FP and FN, in turn improving precision.
Visualization is difficult due to added features.

Confusion matrix with degree = 3
[[81  8]  [ 7 44]]

Less precision, how to choose a proper degree?... see notes.

Note: dimensionality reduction can be performed.
"""

