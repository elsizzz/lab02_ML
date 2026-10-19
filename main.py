# -*- coding: utf-8 -*-
"""Главный пайплайн лабораторной работы №2."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

from config import VARIANT, CONFIG, SEED
from data import load_data, preprocess
from knn import knn_classify, loo_knn
from parzen import parzen_classify, ann_classify, loo_parzen, loo_ann
from potential import potential_classify, loo_potential
from compactness import compactness_profile, greedy_reference_selection
from logistic import train_logistic, evaluate_logistic
from visualization import (
    plot_decision_boundaries,
    plot_compactness_profile,
    plot_reference_pca,
)


def main():
    print("=" * 60)
    print(f"Лабораторная работа №2. Вариант {VARIANT}")
    print("=" * 60)

    # ---------- Часть 1 ----------
    print("\n[Часть 1] Загрузка и предобработка данных...")
    train_df, test_df = load_data()
    X_full, y_full, X_sub, feature_names, scaler, train_names, test_names = preprocess(
        train_df, test_df, CONFIG
    )
    print(f"Размер обучающей выборки: {X_full.shape}, тестовой: {X_sub.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, y_full, test_size=0.3, random_state=SEED, stratify=y_full
    )
    X_val, X_test_local, y_val, y_test_local = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test(local): {X_test_local.shape}")

    # ---------- Часть 2: kNN ----------
    print("\n[Часть 2] kNN классификация...")
    k_max = CONFIG['k_max']
    metric = CONFIG['metric_knn']
    weights = CONFIG['weights_knn']
    p = 3 if metric == 'minkowski' else 2

    best_k, best_acc_loo, accs = loo_knn(X_train, y_train, k_max, metric, weights, p)
    print(f"LOO лучший k = {best_k}, LOO accuracy = {best_acc_loo:.4f}")

    for w in ['uniform', 'distance', 'linear']:
        preds_val = knn_classify(X_train, y_train, X_val, best_k, metric, w, p)
        acc = accuracy_score(y_val, preds_val)
        f1 = f1_score(y_val, preds_val)
        print(f"kNN (k={best_k}, weights={w}): val accuracy={acc:.4f}, f1={f1:.4f}")

    sk_knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric, p=p)
    sk_knn.fit(X_train, y_train)
    preds_sk = sk_knn.predict(X_val)
    print(f"sklearn KNeighborsClassifier: accuracy={accuracy_score(y_val, preds_sk):.4f}, "
          f"f1={f1_score(y_val, preds_sk):.4f}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, k_max + 1), accs, marker='o')
    plt.xlabel('k')
    plt.ylabel('LOO accuracy')
    plt.title('Зависимость LOO accuracy от k')
    plt.grid(True)
    plt.savefig('knn_loo_accuracy.png')
    plt.close()

    # ---------- Часть 3: aNN (окно Парзена) ----------
    print("\n[Часть 3] aNN (окно Парзена)...")
    kernel = CONFIG['kernel_parzen']
    h_max = CONFIG['h_max']
    h_values = np.linspace(0.05, h_max, 20)

    best_h, best_acc_h, accs_h = loo_parzen(X_train, y_train, h_values, kernel, metric, p)
    print(f"Parzen фиксированная ширина: лучший h = {best_h:.4f}, LOO accuracy = {best_acc_h:.4f}")

    preds_parzen = parzen_classify(X_train, y_train, X_val, best_h, kernel, metric, p)
    print(f"Parzen val: accuracy={accuracy_score(y_val, preds_parzen):.4f}, "
          f"f1={f1_score(y_val, preds_parzen):.4f}")

    k_values = range(1, min(20, len(X_train)) + 1)
    best_k_ann, best_acc_ann, accs_ann = loo_ann(X_train, y_train, k_values, kernel, metric, p)
    print(f"aNN переменная ширина: лучший k = {best_k_ann}, LOO accuracy = {best_acc_ann:.4f}")

    preds_ann = ann_classify(X_train, y_train, X_val, best_k_ann, kernel, metric, p)
    print(f"aNN val: accuracy={accuracy_score(y_val, preds_ann):.4f}, "
          f"f1={f1_score(y_val, preds_ann):.4f}")

    print("Визуализация решающих границ на make_moons...")
    X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=SEED)
    X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(
        X_moon, y_moon, test_size=0.3, random_state=SEED
    )
    h_list = [0.1, 0.3, 0.8]
    params_list = [{'h': h, 'kernel': kernel} for h in h_list]
    titles = [f'h={h:.2f}' for h in h_list]

    def parzen_wrapper(X_train, y_train, X_test, h, kernel):
        return parzen_classify(X_train, y_train, X_test, h, kernel,
                               metric='euclidean', p=2)

    plot_decision_boundaries(X_moon_train, y_moon_train, parzen_wrapper,
                             params_list, titles, 'parzen_moons.png')

    # ---------- Часть 4: Профиль компактности и отбор эталонов ----------
    print("\n[Часть 4] Профиль компактности и отбор эталонных объектов...")
    Pi = compactness_profile(X_train, y_train, metric, p, max_m=20)
    plot_compactness_profile(Pi, 'compactness_profile.png')

    mask, accs_ref, sizes_ref = greedy_reference_selection(X_train, y_train, metric, p)
    print(f"Отбор эталонов: начальный размер {X_train.shape[0]}, "
          f"конечный {mask.sum()}, accuracy={accs_ref[-1]:.4f}")

    plot_reference_pca(X_train, mask, y_train, 'reference_pca.png')

    # ---------- Часть 5: Потенциальные функции ----------
    print("\n[Часть 5] Метод потенциальных функций...")
    best_h_pot, best_acc_pot, _ = loo_potential(X_train, y_train, h_values, kernel, metric, p)
    preds_pot = potential_classify(X_train, y_train, X_val, best_h_pot, kernel, metric, p)
    print(f"Потенциальные функции (h={best_h_pot:.4f}): "
          f"accuracy={accuracy_score(y_val, preds_pot):.4f}, f1={f1_score(y_val, preds_pot):.4f}")

    # ---------- Часть 7: Логистическая регрессия ----------
    print("\n[Часть 7] Логистическая регрессия...")
    lr_model, best_C, best_acc_lr = train_logistic(X_train, y_train, seed=SEED)
    print(f"Лучший C = {best_C}, CV accuracy = {best_acc_lr:.4f}")
    acc_lr, f1_lr = evaluate_logistic(lr_model, X_val, y_val)
    print(f"LogisticRegression val: accuracy={acc_lr:.4f}, f1={f1_lr:.4f}")

    # ---------- Часть 8: Submission ----------
    print("\n[Часть 8] Формирование submission.csv...")
    best_preds_sub = knn_classify(X_full, y_full, X_sub, best_k, metric, weights, p)
    submission = pd.DataFrame({'isAlive': best_preds_sub.astype(int)})
    submission.to_csv('submission.csv', index=False)
    print("submission.csv сохранён.")

    print("\nГотово. Все графики сохранены в текущей директории.")


if __name__ == '__main__':
    main()