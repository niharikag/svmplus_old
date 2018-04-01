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
from cvxopt import matrix, solvers
from numpy.matlib import repmat
from numpy import linalg as LA
from base  import BaseSVMPlus

class SVMPlus(six.with_metaclass(ABCMeta, BaseSVMPlus, BaseEstimator)):
    """Base class for SVM plus classification
    """

    #_kernels = ["linear", "poly", "rbf"]

    def __init__(self, C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-5):

        super(SVMPlus, self).__init__(C=1, gamma=1,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-5)


    def fit(self, X, XStar, y):
        """Fit the SVM model according to the given training data.
        """
        #X, y = check_X_y(X, y, 'csr')
        n_samples, n_features = X.shape

        return super(SVMPlus, self).fit(X, XStar, y)



    def project(self, X):
        return super(SVMPlus, self).project(X)+ self.intercept_


    def predict(self, X):
        return np.sign(self.project(X))


    def decision_function(self, X):
        return self.project(self, X)


