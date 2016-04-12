# coding: utf-8
import pandas as pd

import datetime

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier

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

if __name__ == '__main__':
    data = pd.read_csv('data/features.csv', index_col='match_id')

    boosting(data)
