import numpy as np
from svmplus import SVMPlus
from sklearn import svm
from sklearn.datasets import load_digits
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import scipy.ndimage


def testLinearSVMPlus():
    mean1 = np.zeros(2)
    mean2 = np.ones(2)

    cov2 = cov1 = .5 * np.eye(2, 2)

    # cov2 = .15 * np.eye(2,2)
    n = 10
    X1 = np.random.multivariate_normal(mean1, cov=cov1, size=n)
    X2 = np.random.multivariate_normal(mean2, cov=cov2, size=n)

    X = np.vstack((X1, X2))
    y = np.concatenate((np.ones(n), -np.ones(n)))

    X1Star = np.zeros((n, 2))
    X2Star = np.zeros((n, 2))
    i = 0
    for x in X1:
        X1Star[i, 0] = LA.norm(x - mean1)
        X1Star[i, 1] = LA.norm(x - mean2)
        i = i + 1

    i = 0
    for x in X2:
        X2Star[i, 0] = LA.norm(x - mean2)
        X2Star[i, 1] = LA.norm(x - mean1)
        i = i + 1

    XStar = np.vstack((X1Star, X2Star))

    ntest = 50
    # Predict
    X1 = np.random.multivariate_normal(mean1, cov=cov1, size=ntest)
    X2 = np.random.multivariate_normal(mean2, cov=cov2, size=ntest)
    X_test = np.vstack((X1, X2))
    y_test = np.concatenate((np.ones(ntest), -np.ones(ntest)))

    svmp = SVMPlus(C=1000, gamma = .00001, kernel_x="linear",
                   kernel_xstar="linear")
    svmp.fit(X, XStar, y)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


def testPolynomialSVMPlus():
    mean1 = np.zeros(2)
    mean2 = np.ones(2)

    cov2 = cov1 = .5 * np.eye(2, 2)

    # cov2 = .15 * np.eye(2,2)
    n = 20
    X1 = np.random.multivariate_normal(mean1, cov=cov1, size=n)
    X2 = np.random.multivariate_normal(mean2, cov=cov2, size=n)

    X = np.vstack((X1, X2))
    y = np.concatenate((np.ones(n), -np.ones(n)))

    X1Star = np.zeros((n, 2))
    X2Star = np.zeros((n, 2))
    i = 0
    for x in X1:
        X1Star[i, 0] = LA.norm(x - mean1)
        X1Star[i, 1] = LA.norm(x - mean2)
        i = i + 1

    i = 0
    for x in X2:
        X2Star[i, 0] = LA.norm(x - mean2)
        X2Star[i, 1] = LA.norm(x - mean1)
        i = i + 1

    XStar = np.vstack((X1Star, X2Star))

    ntest = 50
    # Predict
    X1 = np.random.multivariate_normal(mean1, cov=cov1, size=ntest)
    X2 = np.random.multivariate_normal(mean2, cov=cov2, size=ntest)
    X_test = np.vstack((X1, X2))
    y_test = np.concatenate((np.ones(ntest), -np.ones(ntest)))

    svmp = SVMPlus(C=1, gamma = 1, kernel_x="poly",
                   kernel_xstar="poly")
    svmp.fit(X, XStar, y)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


def testRbfSVMPlus():
    digits = load_digits(n_class=2)
    X = digits.data
    y = digits.target
    y[y == 0] = -1

    resizedImage = np.zeros((len(X),16))
    for i in range(len(X)):
        originalImage = X[i].reshape(8, 8)
        originalImage = scipy.ndimage.zoom(originalImage, 4 / (8), order=0)
        resizedImage[i] = originalImage.reshape(16,)

    XStar = X[:]
    X = resizedImage

    X_train, X_test, y_train, y_test, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.3)

    print(X_train.shape)

    XStar = XStar[indices_train]
    # train and predict using SVM implemented in sklearn
    clf = svm.SVC(gamma=.0001, C=100.)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(sum(y_predict == y_test), "prediction accuracy using sklearn.svm")

    # train and predict using SVM implemented
    svmp = SVMPlus(C=1000, gamma=.0001, kernel_x="rbf", gamma_x = .00001,
                   kernel_xstar="rbf", gamma_xstar = .00001)
    svmp.fit(X_train, XStar, y_train)
    y_predict = svmp.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


if __name__ == "__main__":
    testLinearSVMPlus()
    testPolynomialSVMPlus()
    testRbfSVMPlus()
