# coding: utf-8
import numpy as np
import pandas as pd

import datetime

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__author__ = 'Adel'


def boosting(dataframe):
    df = dataframe.copy()

    # Выводим первые 5 строк из df
    print(df.head())

    y = df.radiant_win

    # Избавляемся от полей, отсутствующих в тестовой выборке
    delete_cols = ['radiant_win', 'duration', 'tower_status_radiant',
                   'tower_status_dire', 'barracks_status_radiant',
                   'barracks_status_dire']
    df.drop(delete_cols, inplace=True, axis=1)

    # Проверяем есть ли пропуски в данных
    rows = len(df.index)
    data_with_skip = df.count()[df.count() != rows]
    print('Rows with empty values:')
    print(data_with_skip)

    # Количество пустых полей
    print(rows)
    for column in df.columns:
        filled = df[column].count()
        if filled != rows:
            print(column, rows - filled)

    # Заполняем пропуски
    df = df.fillna(0)

    # GradientBoostingClassifier
    for n_est in [10, 20, 30, 40, 50]:
        start_time = datetime.datetime.now()
        kf = KFold(y.size, n_folds=5, shuffle=True)
        clf = GradientBoostingClassifier(n_estimators=n_est)
        clf.fit(df, y)
        scores = cross_validation.cross_val_score(clf, df, y, cv=kf,
                                                  scoring='roc_auc')
        mean = scores.mean()
        t = datetime.datetime.now() - start_time
        print('n_est:{}, mean: {}, time: {}'.format(n_est, mean, t))


def log_regr(dataframe):
    df = dataframe.copy()
    # Выводим первые 5 строк из df
    print(df.head())
    y = df.radiant_win

    # Избавляемся от полей, отсутствующих в тестовой выборке
    delete_cols = ['radiant_win', 'duration', 'tower_status_radiant',
                   'tower_status_dire', 'barracks_status_radiant',
                   'barracks_status_dire']
    df.drop(delete_cols, inplace=True, axis=1)

    # Проверяем есть ли пропуски в данных
    rows = len(df)
    data_with_skip = df.count()[df.count() != rows]
    print('Rows with empty values:')
    print(data_with_skip)

    # Заполняем пропуски
    df = df.fillna(0)
    X = df

    # Delete categorical columns
    del_list = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero',
                'r4_hero','r5_hero', 'd1_hero', 'd2_hero',
                'd3_hero', 'd4_hero', 'd5_hero']
    df.drop(del_list, inplace=True, axis=1)
    X = df

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    kf = KFold(y.size, n_folds=5, shuffle=True)
    c = 0.01
    clf = LogisticRegression(penalty='l2', C=c)
    clf.fit(X, y)
    start_time = datetime.datetime.now()
    scores = cross_validation.cross_val_score(clf, X, y, cv=kf,
                                              scoring='roc_auc')
    print 'Time elapsed:', datetime.datetime.now() - start_time
    mean = scores.mean()
    print(c, mean)

    # Find unique players
    data = pd.read_csv('data/data/features.csv', index_col='match_id')
    heroes_cols_list = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                        'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
    uniq_heroes_list = set()
    for row in heroes_cols_list:
        for id in data[row].unique():
            uniq_heroes_list.add(id)
    print(uniq_heroes_list)
    print('Number of uniq heroes:{}'.format(len(uniq_heroes_list)))

    # Coding information about heroes
    # N — количество различных героев в выборке
    N = 113  # count heroes in heroes.csv
    X_pick = np.zeros((df.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in xrange(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    X_2 = np.concatenate([X, X_pick], axis=1)
    
    # Scaler
    scaler = StandardScaler()
    scaler.fit(X_2)
    X_2 = scaler.transform(X_2)

    kf = KFold(y.size, n_folds=5, shuffle=True)
    c = 0.01
    clf = LogisticRegression(penalty='l2', C=c)
    clf.fit(X_2, y)
    start_time = datetime.datetime.now()
    scores = cross_validation.cross_val_score(clf, X_2, y, cv=kf,
                                              scoring='roc_auc')
    print 'Time elapsed:', datetime.datetime.now() - start_time
    mean = scores.mean()
    print(c, mean)


if __name__ == '__main__':
    data = pd.read_csv('data/features.csv', index_col='match_id')

    boosting(data)
    log_regr(data)
