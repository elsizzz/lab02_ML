# -*- coding: utf-8 -*-
"""Загрузка и предобработка данных GoT."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Колонки, которые дают утечку/шум и не описаны в задании
LEAKY_COLS = [
    'father', 'mother', 'spouse', 'heir', 'successor',
    'allegiance', 'books', 'actor', 'siblings',
]


def load_data():
    train_url = "https://raw.githubusercontent.com/EvADarya/Data_processing-DLS-/refs/heads/main/game_of_thrones_train.csv"
    test_url = "https://raw.githubusercontent.com/EvADarya/Data_processing-DLS-/refs/heads/main/game_of_thrones_test.csv"
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)
    return train, test


def preprocess(train, test, config):
    """Предобработка данных согласно варианту."""
    train = train.copy()
    test = test.copy()
    train_names = train['name'].values if 'name' in train.columns else None
    test_names = test['name'].values if 'name' in test.columns else None

    train['_is_train'] = 1
    test['_is_train'] = 0
    if 'isAlive' not in test.columns:
        test['isAlive'] = np.nan

    df = pd.concat([train, test], ignore_index=True)

    # --- сразу выкидываем «утечные» и служебные колонки ---
    for c in LEAKY_COLS + ['S.No', 'title']:
        if c in df.columns:
            df = df.drop(columns=[c])

    # --- age ---
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())

    # --- dateOfBirth ---
    if 'dateOfBirth' in df.columns:
        df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], errors='coerce')
        df['birth_year'] = df['dateOfBirth'].dt.year
        df['birth_year'] = df['birth_year'].fillna(df['birth_year'].median())
        df = df.drop(columns=['dateOfBirth'])

    # --- culture ---
    if 'culture' in df.columns:
        df['culture'] = df['culture'].fillna('no_data')
        if config['culture'] == 'full':
            culture_map = {
                'Northmen': 'the North',
                'Ironborn': 'the Iron Islands',
                'Valemen': 'the Mountain and the Vale',
                'Rivermen': 'the Isles and Rivers',
                'Westermen': 'the Rock',
                'Stormlanders': 'the Stormlands',
                'Reachmen': 'the Reach',
                'Dornishmen': 'Dorne',
                'Braavosi': 'Essos Nations',
                'Volantene': 'Essos Nations',
                'Myrish': 'Essos Nations',
                'Pentoshi': 'Essos Nations',
                'Lysene': 'Essos Nations',
                'Tyroshi': 'Essos Nations',
                'Ghiscari': 'Essos Nations',
                'Dothraki': 'Essos Nations',
                'Qartheen': 'Essos Nations',
                'Meereenese': 'Essos Nations',
                'Astapori': 'Essos Nations',
                'Yunkai': 'Essos Nations',
                'Naathi': 'Essos Nations',
                'Summer Islander': 'Other Nations',
                'Ibbenese': 'Other Nations',
                'Jogos Nhai': 'Other Nations',
                'Sarnori': 'Other Nations',
                'Andal': 'Other Nations',
                'Rhoynar': 'Other Nations',
                'First Men': 'Old Nations',
                'Children of the Forest': 'Old Nations',
                'Giants': 'Old Nations',
                'Others': 'Other Nations',
                'no_data': 'culture_no_data'
            }
            df['culture'] = df['culture'].map(culture_map).fillna('Other Nations')
        elif config['culture'] == 'rare':
            top_k = 10
            top_cultures = df['culture'].value_counts().nlargest(top_k).index
            df['culture'] = df['culture'].apply(lambda x: x if x in top_cultures else 'other')
        le_culture = LabelEncoder()
        df['culture_enc'] = le_culture.fit_transform(df['culture'].astype(str))
        df = df.drop(columns=['culture'])

    # --- house ---
    if 'house' in df.columns:
        df['house'] = df['house'].fillna('no_data')
        if config['house'] == 'label':
            le_house = LabelEncoder()
            df['house_enc'] = le_house.fit_transform(df['house'].astype(str))
            df = df.drop(columns=['house'])
        elif config['house'] == 'top10':
            top_houses = df['house'].value_counts().nlargest(10).index
            df['house'] = df['house'].apply(lambda x: x if x in top_houses else 'other')
            le_house = LabelEncoder()
            df['house_enc'] = le_house.fit_transform(df['house'].astype(str))
            df = df.drop(columns=['house'])

    # --- popularity ---
    if 'popularity' in df.columns:
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
        if config['popularity'] == 'log':
            M = config.get('M', 100)
            df['popularity'] = np.log1p(df['popularity'] * M + 1)
        elif config['popularity'] == 'qcut':
            n_bins = config.get('qcut_bins', 5)
            df['popularity'] = pd.qcut(df['popularity'], q=n_bins, labels=False, duplicates='drop')

    # name уже выкинули выше через LEAKY_COLS/служебные, но проверим
    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    # One-hot только для оставшихся object-колонок (обычно уже нет)
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Разделяем обратно
    if '_is_train' not in df.columns:
        raise KeyError("'_is_train' пропал до разделения.")
    if 'isAlive' not in df.columns:
        raise KeyError("'isAlive' пропал до разделения.")

    train_df = df[df['_is_train'] == 1].copy().drop(columns=['_is_train'])
    test_df  = df[df['_is_train'] == 0].copy().drop(columns=['_is_train'])

    y_train = train_df['isAlive'].astype(int).values
    X_train = train_df.drop(columns=['isAlive'])
    X_test  = test_df.drop(columns=['isAlive'])

    # --- финальная очистка: NaN / inf ---
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Приводим к float и одинаковому набору колонок
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # --- убираем дубликаты строк по признакам (иначе LOO=1.0) ---
    X_train_np = X_train.values.astype(float)
    y_train_np = y_train

    # Dedup: оставляем только уникальные строки признаков
    _, unique_idx = np.unique(X_train_np, axis=0, return_index=True)
    # Если дубликаты имеют разные метки — оставляем первый вариант,
    # чтобы не «подсовывать» противоречивые данные в 1NN.
    unique_idx = np.sort(unique_idx)
    X_train_np = X_train_np[unique_idx]
    y_train_np = y_train_np[unique_idx]

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test.values.astype(float))

    return X_train_scaled, y_train_np, X_test_scaled, list(X_train.columns), scaler, train_names, test_names