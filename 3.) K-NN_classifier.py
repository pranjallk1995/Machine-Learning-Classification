#K-NN Classifier

#inporting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.4)

#fitting K-NN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier.fit(X_train, Y_train.values.ravel())

#predicting values
Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

#visualising Training Data

fig, (ax1, ax2) = plt.subplots(nrows = 2)

Y_train = Y_train.values
for i in range(0, len(Y_train)):
    if Y_train[i] == 1:
        Y_train[i] = 25
    else:
        Y_train[i] = -25
class_no = np.ma.masked_where(Y_train > 0, Y_train)       # important class_yes and class_no appear to be interchanged.
class_yes = np.ma.masked_where(Y_train < 0, Y_train)

ax1.scatter(X_train[0], X_train[1], marker = 'o', s = abs(class_yes), color = "blue", label = "Buys")
ax1.scatter(X_train[0], X_train[1], marker = 'x', s = abs(class_no), color = "red", label = "Dosen't Buy")
ax1.set_title("Buys SUV Training data")
ax1.set_xlabel("Age")
ax1.set_ylabel("Salary")
ax1.legend()

#Visualizing K-NN.
ngridx = 100
ngridy = 200
x, y = X_test[0].values, X_test[1].values
z = classifier.predict(np.array([x, y]).T)

xi = np.linspace(min(x), max(x), ngridx)
yi = np.linspace(min(y), max(y), ngridy)
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

cntr2 = ax2.contourf(xi, yi, zi, levels = 10, cmap="RdBu", alpha = 0.4)

cls = Y_test.values
class_yes = []
class_no = []
for i in range(0, len(cls)):
    if cls[i] == 1:
        class_yes.append(25)
        class_no.append(0)
    else:
        class_yes.append(0)
        class_no.append(25)

ax2.scatter(x, y, marker = 'o', s = class_yes, color = 'blue', label = "Buys")
ax2.scatter(x, y, marker = 'x', s = class_no, color = 'red', label = "Dosen't Buy")
ax2.axis((min(x)-0.2, max(x)+0.2, min(y)-0.2, max(y)+0.2))
ax2.set_title('Buys SUV Test data with K-NN classification')
ax2.set_xlabel("Age")
ax2.set_ylabel("Salary")
ax2.legend()

plt.show()

#performs slightly better than improved Logistic Regression on this dataset. (see confusion matrix).