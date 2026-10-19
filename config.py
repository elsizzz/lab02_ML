# -*- coding: utf-8 -*-
"""Конфигурация варианта."""

VARIANT = 1  # номер варианта
SEED = 1     # random_state

def get_variant_config(variant):
    """Возвращает параметры для выбранного варианта.
    Здесь приведён только вариант 1. Добавьте остальные по таблице.
    """
    configs = {
        1: {
            'seed': 1,
            'k_max': 15,
            'h_max': 1.0,
            'metric_knn': 'euclidean',
            'weights_knn': 'uniform',
            'kernel_parzen': 'gaussian',
            'popularity': 'log',
            'M': 100,
            'culture': 'full',
            'house': 'label',
            'n_trees': 5,
            'n_neighbors_ann': 5,
            'query_character': 'Jon Snow',
        },
        # Добавьте другие варианты при необходимости
    }
    return configs.get(variant, configs[1])

CONFIG = get_variant_config(VARIANT)