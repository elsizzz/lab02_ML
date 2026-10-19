# -*- coding: utf-8 -*-
"""Окно Парзена: фиксированная и переменная ширина."""

import numpy as np
from metrics import get_distance_matrix


def kernel_func(r, kernel):
    if kernel == 'gaussian':
        return np.exp(-2 * r ** 2)
    elif kernel == 'rectangular':
        return (np.abs(r) <= 1).astype(float)
    elif kernel == 'triangular':
        return np.maximum(0, 1 - np.abs(r))
    elif kernel == 'epanechnikov':
        return np.maximum(0, 1 - r ** 2)
    elif kernel == 'quartic':
        return np.maximum(0, 1 - r ** 2) ** 2
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def parzen_classify(X_train, y_train, X_test, h, kernel, metric='euclidean', p=2):
    """Окно Парзена с фиксированной шириной h."""
    dists = get_distance_matrix(X_train, X_test, metric, p)
    n_test = X_test.shape[0]
    preds = np.zeros(n_test)
    for i in range(n_test):
        r = dists[i] / h
        K = kernel_func(r, kernel)
        score = np.sum(K * y_train)
        preds[i] = 1 if score >= 0.5 * np.sum(K) else 0
    return preds


def ann_classify(X_train, y_train, X_test, k, kernel, metric='euclidean', p=2):
    """aNN: переменная ширина окна = расстояние до k-го соседа."""
    dists = get_distance_matrix(X_train, X_test, metric, p)
    n_test = X_test.shape[0]
    preds = np.zeros(n_test)
    for i in range(n_test):
        sorted_idx = np.argsort(dists[i])
        h = dists[i][sorted_idx[k - 1]]
        if h == 0:
            h = 1e-6
        r = dists[i] / h
        K = kernel_func(r, kernel)
        score = np.sum(K * y_train)
        preds[i] = 1 if score >= 0.5 * np.sum(K) else 0
    return preds


def loo_parzen(X, y, h_values, kernel, metric='euclidean', p=2):
    """LOO-подбор h для Parzen."""
    dists = get_distance_matrix(X, X, metric, p)
    np.fill_diagonal(dists, np.inf)
    n = X.shape[0]
    best_h, best_acc = h_values[0], 0
    accs = []
    for h in h_values:
        correct = 0
        for i in range(n):
            r = dists[i] / h
            K = kernel_func(r, kernel)
            score = np.sum(K * y)
            pred = 1 if score >= 0.5 * np.sum(K) else 0
            if pred == y[i]:
                correct += 1
        acc = correct / n
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_h = h
    return best_h, best_acc, accs


def loo_ann(X, y, k_values, kernel, metric='euclidean', p=2):
    """LOO-подбор k для aNN."""
    dists = get_distance_matrix(X, X, metric, p)
    np.fill_diagonal(dists, np.inf)
    n = X.shape[0]
    best_k, best_acc = k_values[0], 0
    accs = []
    for k in k_values:
        correct = 0
        for i in range(n):
            sorted_idx = np.argsort(dists[i])
            h = dists[i][sorted_idx[k - 1]]
            if h == 0:
                h = 1e-6
            r = dists[i] / h
            K = kernel_func(r, kernel)
            score = np.sum(K * y)
            pred = 1 if score >= 0.5 * np.sum(K) else 0
            if pred == y[i]:
                correct += 1
        acc = correct / n
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k, best_acc, accs