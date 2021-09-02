import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def get_Adult_Data():
    os.chdir("data")

    (train := pd.read_csv("adult.data", header=None, na_values= ' ?').dropna()).drop([2, 3, 13], axis=1, inplace=True)
    (test := pd.read_csv("adult.test", header=None, na_values= ' ?').dropna()).drop([2, 3, 13], axis=1, inplace=True)
    
    X_trn = np.array([i.toarray()[0] for i in  ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 4, 5, 6, 7])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(train)[:, :-1])])
    Y_trn = LabelEncoder().fit_transform(np.array(train)[:, -1])
    X_tst = np.array([i.toarray()[0] for i in ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 4, 5, 6, 7])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(test)[:, :-1])])
    Y_tst = LabelEncoder().fit_transform(np.array(test)[:, -1])
    return X_trn, X_tst, Y_trn, Y_tst

def get_German_Data(seed=0):
    os.chdir("data")
    data = pd.read_csv("german.data", sep=" ")

    X = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(data)[:, :-1])
    Y = LabelEncoder().fit_transform(np.array(data)[:, -1])

    return train_test_split(X, Y, test_size=0.33, random_state=seed)