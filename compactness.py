# -*- coding: utf-8 -*-
"""Профиль компактности и отбор эталонных объектов."""

import numpy as np
from metrics import get_distance_matrix


def compactness_profile(X, y, metric='euclidean', p=2, max_m=20):
    """Профиль компактности Pi(m).
    Предвычисляем argsort один раз, дальше всё векторно.
    """
    n = X.shape[0]
    dists = get_distance_matrix(X, X, metric, p)
    np.fill_diagonal(dists, np.inf)
    sorted_nn = np.argsort(dists, axis=1)  # (n, n) — делается один раз
    y_arr = np.asarray(y)

    Pi = np.zeros(max_m)
    for m in range(1, max_m + 1):
        mth = sorted_nn[:, m - 1]                       # (n,) — m-й сосед каждого объекта
        Pi[m - 1] = np.mean(y_arr == y_arr[mth])        # векторно
    return Pi


def greedy_reference_selection(X, y, metric='euclidean', p=2,
                               max_removals=100, verbose=True):
    """Жадное удаление эталонов для 1NN с критерием LOO.

    Оптимизация:
      - dists и sorted_nn считаются один раз;
      - first_nn поддерживается инкрементально: при удалении j обновляются
        только те объекты, у которых j был первым соседом;
      - LOO-accuracy считается векторно.
    """
    n = X.shape[0]
    dists = get_distance_matrix(X, X, metric, p)
    np.fill_diagonal(dists, np.inf)
    sorted_nn = np.argsort(dists, axis=1)  # (n, n)
    y_arr = np.asarray(y)

    active = np.ones(n, dtype=bool)

    def find_nn(i):
        """Первый активный сосед i (сам i пропускается)."""
        for j in sorted_nn[i]:
            if j != i and active[j]:
                return j
        return -1

    # Инициализация first_nn
    first_nn = np.array([find_nn(i) for i in range(n)], dtype=int)

    def loo_acc():
        valid = first_nn >= 0
        if not valid.any():
            return 0.0
        correct = np.sum(y_arr[valid] == y_arr[first_nn[valid]])
        return correct / active.sum()

    accs = [loo_acc()]
    sizes = [int(active.sum())]

    for it in range(max_removals):
        active_ids = np.where(active)[0]
        best_acc = -1.0
        best_j = -1

        for j in active_ids:
            active[j] = False
            affected = np.where(first_nn == j)[0]
            backup = first_nn[affected].copy()
            for i in affected:
                first_nn[i] = find_nn(i)
            acc = loo_acc()
            if acc > best_acc:
                best_acc = acc
                best_j = j
            # откат
            first_nn[affected] = backup
            active[j] = True

        if best_acc >= accs[-1]:
            active[best_j] = False
            affected = np.where(first_nn == best_j)[0]
            for i in affected:
                first_nn[i] = find_nn(i)
            accs.append(best_acc)
            sizes.append(int(active.sum()))
            if verbose and (it + 1) % 10 == 0:
                print(f"  [greedy] iter={it+1}, size={sizes[-1]}, acc={best_acc:.4f}")
        else:
            if verbose:
                print(f"  [greedy] останов на iter={it+1}: acc не улучшается")
            break

    return active, accs, sizes