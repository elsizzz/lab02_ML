# -*- coding: utf-8 -*-
"""Метод потенциальных функций (то же, что Parzen с фиксированным h)."""

from parzen import parzen_classify, loo_parzen


def potential_classify(X_train, y_train, X_test, h, kernel, metric='euclidean', p=2):
    return parzen_classify(X_train, y_train, X_test, h, kernel, metric, p)


def loo_potential(X, y, h_values, kernel, metric='euclidean', p=2):
    return loo_parzen(X, y, h_values, kernel, metric, p)