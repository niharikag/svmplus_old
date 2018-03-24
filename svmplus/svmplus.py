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


class SVMPlus(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for SVM plus classification
    """

    _kernels = ["linear", "poly", "rbf"]

    def __init__(self, C, gamma,
                 kernel_x = 'rbf', degree_x = 3, gamma_x ='auto',
                 kernel_xstar = 'rbf', degree_xstar = 3, gamma_xstar ='auto',
                 tol = 1e-5
                 #C, nu, epsilon, shrinking, probability, cache_size,
                 #class_weight, verbose, max_iter, random_state
                 ):


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

        P1 = np.concatenate((matrix(np.outer(y, y) * K) + KStar / float(self.gamma),
                             KStar / float(self.gamma)), axis=1)
        P2 = np.concatenate((KStar / float(self.gamma), KStar / float(self.gamma)), axis=1)
        P = np.concatenate((P1, P2), axis=0)
        A = np.concatenate((np.ones((1, 2 * n_samples)),
                            np.concatenate((np.transpose(matrix(y)), np.zeros((1, n_samples))), axis=1)),
                           axis=0)
        b = np.array([[n_samples * self.C], [0]])
        G = -np.eye(2 * n_samples)
        h = np.zeros((2 * n_samples, 1))

        Q = repmat(sum(KStar + np.transpose(KStar)), 1, 2) * (
                - self.C / float(2 * self.gamma)) - \
            np.concatenate((np.ones((1, n_samples)), np.zeros((1, n_samples))), axis=1)
        q = np.transpose(Q)

        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'),
                         matrix(A, tc='d'), matrix(b, tc='d'))

        # Lagrange multipliers
        alpha = np.ravel(sol['x'][0:n_samples])

        # Support vectors have non zero lagrange multipliers
        sv = alpha > self.tol  # tolerance
        ind = np.arange(len(alpha))[sv]
        alpha = alpha[sv]
        # print("%d support vectors out of %d points" % (len(alpha), n_samples))
        sv_x = X[sv]
        sv_y = y[sv]

        bias = 0

        for n in range(len(alpha)):
            bias += sv_y[n] - np.sum(alpha * sv_y * K[ind[n], sv])

        bias /= len(alpha)

        self.support_vectors_ = sv_x  # support vector's features
        self.support_y_ = sv_y  # support vector's labels
        self.dual_coef_ = alpha
        self.intercept_ = bias



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
            for a, sv_y, sv in zip(self.dual_coef_, self.support_y_, self.support_vectors_):
                s += a * sv_y * kernel_method(X[i], sv, kernel_param)
            y_predict[i] = s
        return y_predict + self.intercept_


    def predict(self, X):
        return np.sign(self.project(X))


    def decision_function(self, X):
        return self.project(self, X)


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
