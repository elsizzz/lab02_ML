# -*- coding: utf-8 -*-
"""Логистическая регрессия с L2-регуляризацией (совместимо со sklearn 1.8+)."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score


def _make_lr(C, seed):
    """Создаёт LogisticRegression c L2-регуляризацией без устаревшего penalty."""
    try:
        # Новый API (sklearn >= 1.8): l1_ratio=0 → чистый L2
        return LogisticRegression(
            C=C, l1_ratio=0, solver='liblinear',
            random_state=seed, max_iter=1000
        )
    except TypeError:
        # Старый API (sklearn < 1.8)
        return LogisticRegression(
            C=C, penalty='l2', solver='liblinear',
            random_state=seed, max_iter=1000
        )


def train_logistic(X_train, y_train, C_values=(0.01, 0.1, 1, 10), seed=42):
    best_C, best_acc = C_values[0], 0
    for C in C_values:
        lr = _make_lr(C, seed)
        scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
        acc = scores.mean()
        if acc > best_acc:
            best_acc = acc
            best_C = C
    model = _make_lr(best_C, seed)
    model.fit(X_train, y_train)
    return model, best_C, best_acc


def evaluate_logistic(model, X_val, y_val):
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds), f1_score(y_val, preds)