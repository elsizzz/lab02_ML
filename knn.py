# -*- coding: utf-8 -*-
"""Собственный kNN и LOO-подбор k."""

import numpy as np
from metrics import get_distance_matrix


def knn_classify(X_train, y_train, X_test, k, metric='euclidean', weights='uniform', p=2):
    """Собственный kNN."""
    dists = get_distance_matrix(X_train, X_test, metric, p)
    n_test = X_test.shape[0]
    preds = np.zeros(n_test)
    for i in range(n_test):
        idx = np.argsort(dists[i])[:k]
        labels = y_train[idx]
        d = dists[i][idx]
        if weights == 'uniform':
            w = np.ones(k)
        elif weights == 'distance':
            w = 1.0 / (d + 1e-10)
        elif weights == 'linear':
            w = (k + 1 - np.arange(1, k + 1)) / k
        elif weights == 'exponential':
            q = 0.5
            w = q ** np.arange(k)
        else:
            raise ValueError(f"Unknown weights: {weights}")
        score = np.sum(w * labels)
        preds[i] = 1 if score >= 0.5 * np.sum(w) else 0
    return preds


def loo_knn(X, y, k_max, metric, weights, p=2):
    """LOO-подбор k для kNN."""
    n = X.shape[0]
    dists = get_distance_matrix(X, X, metric, p)
    np.fill_diagonal(dists, np.inf)
    best_k, best_acc = 1, 0
    accs = []
    for k in range(1, k_max + 1):
        correct = 0
        for i in range(n):
            idx = np.argsort(dists[i])[:k]
            labels = y[idx]
            d = dists[i][idx]
            if weights == 'uniform':
                w = np.ones(k)
            elif weights == 'distance':
                w = 1.0 / (d + 1e-10)
            elif weights == 'linear':
                w = (k + 1 - np.arange(1, k + 1)) / k
            elif weights == 'exponential':
                q = 0.5
                w = q ** np.arange(k)
            score = np.sum(w * labels)
            pred = 1 if score >= 0.5 * np.sum(w) else 0
            if pred == y[i]:
                correct += 1
        acc = correct / n
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k, best_acc, accs