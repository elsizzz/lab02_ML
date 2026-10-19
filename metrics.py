# -*- coding: utf-8 -*-
"""Метрики расстояния."""

import numpy as np


def minkowski_distance_matrix(X, Y, p=2):
    """Матрица расстояний между X (m,d) и Y (n,d) по Минковскому."""
    X = np.array(X)
    Y = np.array(Y)
    diff = np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :])
    return np.sum(diff ** p, axis=2) ** (1.0 / p)


def get_distance_matrix(X_train, X_test, metric, p=2):
    if metric == 'euclidean':
        return minkowski_distance_matrix(X_test, X_train, p=2)
    elif metric == 'manhattan':
        return minkowski_distance_matrix(X_test, X_train, p=1)
    elif metric == 'minkowski':
        return minkowski_distance_matrix(X_test, X_train, p=p)
    else:
        raise ValueError(f"Unknown metric: {metric}")