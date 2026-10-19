# -*- coding: utf-8 -*-
"""Визуализация."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_decision_boundaries(X, y, model_func, params_list, titles, filename):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.figure(figsize=(15, 4))
    for i, (params, title) in enumerate(zip(params_list, titles)):
        plt.subplot(1, len(params_list), i + 1)
        Z = model_func(X, y, np.c_[xx.ravel(), yy.ravel()], **params)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_compactness_profile(Pi, filename='compactness_profile.png'):
    plt.figure()
    plt.plot(range(1, len(Pi) + 1), Pi, marker='o')
    plt.xlabel('m')
    plt.ylabel('Pi(m)')
    plt.title('Профиль компактности')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_reference_pca(X, mask, y, filename='reference_pca.png'):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[~mask, 0], X_pca[~mask, 1], c='lightgray', label='удалённые')
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=y[mask], cmap='bwr',
                edgecolors='k', label='эталоны')
    plt.legend()
    plt.title('Эталонные объекты (PCA)')
    plt.savefig(filename)
    plt.close()