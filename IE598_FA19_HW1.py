# importing the iris datasets
# and of course renaming each parametres for future use
from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# testing if we are getting the correct dataset plus knowing
# the approximate shape of the dataset
print (X_iris.shape, y_iris.shape)
print (X_iris[0], y_iris[0])
# Note that for the "target" set
# we are using 0: setosa, 1: versicolor, and 2: virginic
# and for the "data" set, each variable stands for
# basic informations(aka. features) of each Iris flowers

from sklearn.model_selection import train_test_split
# from the textbook we were required to put
# "from sklearn.cross_validation import train_test_split"
# but it showed as no such model existed
# searched online and then getting should be using
# sklearn.model_slection istead of sklearn.cross_validation
# The train_test_split function automatically builds the training and evaluation datasets, 
# randomly selecting the samples. 
# Why not just select the first 112 examples? 
# This is because it could happen that the instance ordering within the sample could matter 
# and that the first instances could be different to the last ones
from sklearn import preprocessing
# pre-processing refers to the transformations applied to your data
# before feeding it to the algorithm
X, y = X_iris[:, :2], y_iris
# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print (X_train.shape, y_train.shape)
# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
# using this part to draw the plot
colors = ['red', 'greenyellow', 'blue']
# giving definations to colors
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
    # The scatter function simply plots the first feature value (sepal width) 
    # for each instance versus its second feature value (sepal length)
    # and uses the target class values to assign a different color for each class. 
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
# Giving the plot name and names for x-axis and y-axis

from sklearn.linear_model import SGDClassifier
# No module named 'sklearn.linear_modelsklearn'
# so we are using sklearn.linear_model
# I guess this is another version bugs that we need to fix
clf = SGDClassifier()
clf.fit(X_train, y_train)
# clf = SGDClassifier() aka. one of those algorithms you can never understand

print (clf.coef_)
print (clf.intercept_)


import numpy as np
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
Xs = np.arange(x_min, x_max, 0.5)
# creates an instance of ndarray with evenly spaced values 
# and returns the reference to it
fig, axes = plt.subplots(1, 3)
# subplot() function can be called to plot two or more plots in one figure
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
     axes[i].set_aspect('equal')
     axes[i].set_title('Class '+ str(i) + ' versus the rest')
     axes[i].set_xlabel('Sepal length')
     axes[i].set_ylabel('Sepal width')
     axes[i].set_xlim(x_min, x_max)
     axes[i].set_ylim(y_min, y_max)
     plt.sca(axes[i])
     plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
     ys = (-clf.intercept_[i] - Xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
     plt.plot(Xs, ys)
     # since "hold" was deprecated, plot behaves as if hold = True,
     # so you can leave out specifying it explicitly
print (clf.predict(scaler.transform([[4.7, 3.1]])))
print (clf.decision_function(scaler.transform([[4.7, 3.1]])))  
from sklearn import metrics
y_train_pred = clf.predict(X_train)
print (metrics.accuracy_score(y_train, y_train_pred))
y_pred = clf.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))
print (metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
print (metrics.confusion_matrix(y_test, y_pred))

print("My name is {Qianyi Liu}")
print("My NetID is: {qianyil2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



