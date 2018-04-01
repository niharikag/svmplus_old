import numpy as np
from svmplus import SVMPlus
from sklearn import svm
from sklearn.datasets import load_digits
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import scipy.ndimage
import gridSearchCV
import libsvmplus as libsvm


def prepareDigitData():
    digits = load_digits(n_class=2)
    X = digits.data
    y = digits.target
    y[y == 0] = -1

    resizedImage = np.zeros((len(X), 16))
    for i in range(len(X)):
        originalImage = X[i].reshape(8, 8)
        originalImage = scipy.ndimage.zoom(originalImage, 4 / (8), order=0)
        resizedImage[i] = originalImage.reshape(16, )

    XStar = X[:]
    X = resizedImage

    X_train, X_test, y_train, y_test, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.3)

    XStar = XStar[indices_train]
    return X_train, X_test, y_train, y_test, XStar

def testLinearSVMPlus():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    svmp = SVMPlus(C=1000, gamma = .00001, kernel_x="linear",
                   kernel_xstar="linear")
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


def testPolynomialSVMPlus():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    svmp = SVMPlus(C=10, gamma = .01, kernel_x="poly",
                   kernel_xstar="poly")
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

def testGridSerachCV():
    param_grid = {'C': [10, 100],
                  'gamma': [0.0001, 0.001],
                  'gamma_x': [0.0001, 0.001],
                  'gamma_xstar': [0.0001, 0.001]}

    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    gridSearchCV.gridSearchSVMPlus(X_train, y_train, XStar,
                                   param_grid, n_splits=5, logfile=None)


def testRbfSVMPlus():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    # train and predict using SVM plus
    svmp = SVMPlus(C=1000, gamma=.0001, kernel_x="rbf", gamma_x = .00001,
                   kernel_xstar="rbf", gamma_xstar = .00001)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


def testLibSVM():
    X_train, X_test, y_train, y_test, XStar = prepareDigitData()

    svmp =libsvm.libSVMPlus(C=10, gamma=.00001, kernel_x="linear",
                            gamma_x = .00001, kernel_xstar="linear", gamma_xstar = .0001)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

if __name__ == "__main__":
    #testLinearSVMPlus()
    #testPolynomialSVMPlus()
    #testRbfSVMPlus()

    testLibSVM()

    # testGridSerachCV()