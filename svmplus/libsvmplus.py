#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiclass SVM Plus Classification(SVPC)
https://github.com/TBD
niharika gauraha<niharika.gauraha@farmbio.uu.se>
Ola Spjuth<ola.spjuth@farmbio.uu.se>
Learns a binary classifier based on SVM Plus:
LUPI paradigm. Uses scikit learn.
Licensed under a Creative Commons Attribution-NonCommercial 4.0
International License.
Based on SVM+ by Vapnik et al.
"""


import six
from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from numpy import linalg as LA
from sklearn import svm



class libSVMPlus(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for SVM plus classification
    """

    #_kernels = ["linear", "poly", "rbf"]

    def __init__(self, C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-5):


        if gamma == 0 or gamma_x == 0 or gamma_xstar == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        self.C = C
        self.gamma = gamma
        self.kernel_x = kernel_x
        self.degree_x = degree_x
        self.gamma_x = gamma_x
        self.kernel_xstar = kernel_xstar
        self.degree_xstar = degree_xstar
        self.gamma_xstar = gamma_xstar
        self.tol = tol

    def fit(self, X, XStar, y):
        """Fit the SVM model according to the given training data.
        """
        #X, y = check_X_y(X, y, 'csr')
        n_samples, n_features = X.shape


        if self.kernel_x == "linear":
            kernel_method = self._linear_kernel
            kernel_param = None
        elif self.kernel_x == "poly":
            kernel_method = self._poly_kernel
            kernel_param = self.degree_x
        else:
            kernel_method = self._rbf_kernel
            if self.gamma_x == 'auto':
                self.gamma_x = 1 / n_features
            kernel_param = self.gamma_x

        if self.kernel_xstar == "linear":
            kernel_method_star = self._linear_kernel
            kernel_param_star = None
        elif self.kernel_xstar == "poly":
            kernel_method_star = self._poly_kernel
            kernel_param_star = self.degree_xstar
        else:
            kernel_method_star = self._rbf_kernel
            if self.gamma_xstar == 'auto':
                self.gamma_xstar = 1 / XStar.shape[1]
            kernel_param_star = self.gamma_xstar

        # compute the matrix K and KStar (n_samples X n_samples) using kernel function
        K = np.zeros((n_samples, n_samples))
        KStar = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = kernel_method(X[i, :], X[j, :], kernel_param)
                KStar[i, j] = kernel_method_star(XStar[i, :], XStar[j, :], kernel_param_star)

        #n_samples = length(labels);
        #n_class = 2 #unique(labels);

        # append bias
        K = K + 1
        KStar = KStar + 1

        G = np.eye(n_samples) - np.linalg.inv(np.eye(n_samples) + (self.C / self.gamma) * KStar)
        G = (1 / self.C) * G
        H = (K * np.outer(y, y))
        Q = H + G
        #D = np.transpose(range(n_samples))
        #prob = np.concatenate((D, Q), axis = 1)
        model = svm.libsvm.fit(Q, np.ones(n_samples), svm_type=2, nu = 1 / n_samples,
                               kernel = 'precomputed', tol = 1e-5)

        # print("%d support vectors out of %d points" % (len(alpha), n_samples))
        sv_x = X[model[0]]
        sv_y = y[model[0]]

        self.support_vectors_ = sv_x  # support vector's features
        self.support_y_ = sv_y  # support vector's labels
        coeff = model[3]
        self.dual_coef_ = abs(coeff)



    def project(self, X):
        if self.kernel_x == "linear":
            kernel_method = self._linear_kernel
            kernel_param = None
        elif self.kernel_x == "poly":
            kernel_method = self._poly_kernel
            kernel_param = self.degree_x
        else:
            kernel_method = self._rbf_kernel
            kernel_param = self.gamma_x

        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.dual_coef_[0], self.support_y_, self.support_vectors_):
                s += a * sv_y * (kernel_method(X[i], sv, kernel_param) + 1)
            y_predict[i] = s
        return y_predict


    def predict(self, X):
        return np.sign(self.project(X))


    def decision_function(self, X):
        return self.project(X)


    @staticmethod
    # Linear kernel
    def _linear_kernel(x1, x2, param = None):
        return np.dot(x1, x2)


    @staticmethod
    # Polynomial kernel
    def _poly_kernel(x1, x2, param = 2):
        return (1 + np.dot(x1, x2)) ** param


    @staticmethod
    # Radial basis kernel
    def _rbf_kernel(x1, x2, param = 1.0):
        return np.exp(-(LA.norm(x1 - x2) ** 2) * param)