import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.compose import ColumnTransformer


def getData():
    os.chdir("Explain/data")

    (train := pd.read_csv("adult.data", header=None, na_values= ' ?').dropna()).drop([2, 3, 13], axis=1, inplace=True)
    (test := pd.read_csv("adult.test", header=None, na_values= ' ?').dropna()).drop([2, 3, 13], axis=1, inplace=True)
    
    X_trn = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 4, 5, 6, 7])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(train)[:, :-1])
    Y_trn = LabelEncoder().fit_transform(np.array(train)[:, -1])
    X_tst = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 4, 5, 6, 7])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(test)[:, :-1])
    Y_tst = LabelEncoder().fit_transform(np.array(test)[:, -1])
    return X_trn, Y_trn, X_tst, Y_tst