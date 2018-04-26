import numpy as np
from svmplus import SVMPlus
from sklearn import svm
from sklearn.datasets import load_digits
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import scipy.ndimage
import gridSearchCV
import libsvmplus as libsvm
import csv as csv
from sklearn.model_selection import train_test_split

def loadMNISTData():
    ifile = open("../data/mnistResizedData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) for val in row]
        a.append(newRow)
    ifile.close()
    X = np.array([x for x in a]).astype(float)

    ifile = open("../data/mnistData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) for val in row]
        a.append(newRow)
    ifile.close()
    XStar = np.array([x for x in a]).astype(float)

    ifile = open("../data/mnistLabel.csv")
    reader = csv.reader(ifile)
    a = []
    c = 0
    for row in reader:
        c = c+1
        a.append(row)
    ifile.close()
    y = np.array(a).astype(float).reshape(1, c)
    y = np.array([x for x in y]).astype(int)
    y = y[0]


    X_train, X_test, y_train, y_test, indices_train, indices_test = \
        train_test_split(X, y, range(len(X)), test_size=0.5,
                     stratify=y, random_state=7)
    XStar = XStar[indices_train]

    return X_train, X_test, y_train, y_test, XStar


X_train, X_test, y_train, y_test, XStar = loadMNISTData()



svmp = SVMPlus(C=1, gamma=1, kernel_x= "rbf", gamma_x = .01,
                   kernel_xstar="rbf", gamma_xstar = 0.01)
svmp.fit(X_train, XStar, y_train)
y_predict = svmp.predict(X_test)
correct = np.sum(y_predict == y_test)
print("Prediction accuracy of SVM")
print("%d out of %d predictions correct" % (correct, len(y_predict)))






















'''
import h5py
import scipy.io
test = scipy.io.loadmat('../data/mnist_plus.mat')

f = h5py.File('../data/mnist_plus.mat','r')

for item in f.attrs.keys():
    print(item + ":", f.attrs[item])

print(type(f))
data = f.get('data')
data = np.array(data) # For converting to numpy array
print(data.shape)
print(data)
f.close()
'''

