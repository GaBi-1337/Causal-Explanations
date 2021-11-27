import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

os.chdir("data")

class Representer:

    def __init__(self, data):
        self.Data = data
        self.categorical = list(set(data.columns) - set(data._get_numeric_data()))
        self.mappings = dict()
        self.newData = pd.DataFrame(np.zeros(len(data)), columns=['temp'])
        self.onehot = dict()
        self.label = dict()
        for col in data.columns:
            if col in self.categorical:
                values = set(np.array(data[col]))
                representation = list()
                self.onehot[col] = ohe = np.array(OneHotEncoder(drop='first').fit_transform(np.array(data[col]).reshape(-1, 1)).toarray(), dtype=int)
                self.label[col] = le = np.array(LabelEncoder().fit_transform(np.array(data[col])), dtype=int)
                for i in range(len(data)):
                    if data[col][i] in values:
                        representation.append((data[col][i], np.array(ohe[i]), le[i]))
                        values.remove(data[col][i])
                    elif len(values) == 0:
                        break
                self.mappings[col] = representation
                self.newData = pd.concat([self.newData, pd.DataFrame(ohe)], axis=1)
            else:
                self.newData = pd.concat([self.newData, data[col]], axis=1)
        del self.newData['temp']

    def get_data(self, X=None, Y=None, N=None):
        if X == None or Y == None:
            return np.array(self.newData, dtype=int) if N == None else np.array(self.newData, dtype=int)[: N]
        else:
            N = len(self.Data) if N==None else N
            newData = pd.DataFrame(np.zeros(N), columns=['temp'])
            for col in X:
                if col in self.categorical:
                    newData = pd.concat([newData, pd.DataFrame(self.onehot[col][: N])], axis=1)
                else:
                    newData = pd.concat([newData, self.Data[col][: N]], axis=1)
            if Y in self.categorical:
                newData = pd.concat([newData, pd.DataFrame(self.label[Y][: N])], axis=1)
            else:
                newData = pd.concat([newData, self.Data[Y][: N]], axis=1)
            del newData['temp']
            return np.array(newData, dtype=int)
    
    def ohe_transform(self, label, column):
        for value in self.mappings[column]:
            if value[0] == label:
                return value[1]

    def le_transform(self, label, column):
        for value in self.mappings[column]:
            if value[0] == label:
                return value[2]
    
    def point_transform(self, vector, X=None, Y=None):
        newPoint = []
        for col in self.Data.columns[: -1]:
            if X is None or col in X:
                if col in self.categorical:
                    newPoint += list(self.ohe_transform(vector[col], col))
                else:
                    newPoint += [vector[col]]
        if Y != None:
            newPoint += [self.le_transform(vector[Y], Y)]
        return np.array(newPoint, dtype=int)
    
    def ohe_inverse(self, vector, column):
        for value in self.mappings[column]:
            if np.array_equal(value[1], vector):
                return value[0]
    
    def le_inverse(self, label, column):
        for value in self.mappings[column]:
            if value[2] == label:
                return value[0]
    
    def point_inverse(self, vector, y_present=False):
        newPoint = []
        offset = 0 
        for col in self.Data.columns[: -1]:
            if col in self.categorical:
                col_len = len(self.mappings[col][0][1])
                newPoint += [self.ohe_inverse(vector[col + offset: col + offset +  col_len], col)]
                offset += col_len - 1
            else:
                newPoint += [vector[col + offset]]
        if y_present:
            newPoint += [self.le_inverse(vector[-1], col + 1 )]
        return np.array(newPoint)
        

def get_Adult_Data():

    train = pd.read_csv("adult.data", header=None, na_values= ' ?')
    test = pd.read_csv("adult.test", header=None, na_values= ' ?')
    
    train = train.dropna()
    test = test.dropna()

    train.drop([2, 3, 13], axis=1, inplace=True)
    test.drop([2, 3, 13], axis=1, inplace=True)

    X_trn = np.array([i.toarray()[0] for i in  ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 4, 5, 6, 7])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(train)[:, :-1])])
    Y_trn = LabelEncoder().fit_transform(np.array(train)[:, -1])
    X_tst = np.array([i.toarray()[0] for i in ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [1, 3, 4, 5, 6, 7])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(test)[:, :-1])])
    Y_tst = LabelEncoder().fit_transform(np.array(test)[:, -1])
    return X_trn, X_tst, Y_trn, Y_tst

def get_German_Data(seed=0):
    data = pd.read_csv("german.data", sep=" ")
    
    X = ColumnTransformer([('one_hot_encoder', OneHotEncoder(drop='first'), [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])], remainder='passthrough', n_jobs=-1).fit_transform(np.array(data)[:, :-1])
    Y = LabelEncoder().fit_transform(np.array(data)[:, -1])

    return train_test_split(X, Y, test_size=0.33, random_state=seed)

def get_ACS_Data(seed=0):
    data = np.array(pd.read_csv("acs13.csv"))

    X = np.array(data[:, :-1])
    Y = np.array(data[:, -1])
    
    return train_test_split(X, Y, test_size=0.33, random_state=seed)

if __name__ == '__main__':
    train = pd.read_csv("adult.data", header=None, na_values= ' ?')
    test = pd.read_csv("adult.test", header=None, na_values= ' ?') 
    train = train.dropna()
    test = test.dropna()
    train.drop([2, 3, 13], axis=1, inplace=True)
    test.drop([2, 3, 13], axis=1, inplace=True)
    data = pd.concat([train, test], ignore_index=True)
    data = data.rename(columns={key: value for value, key in enumerate(data.columns)})
    rep = Representer(data)
    point = rep.get_data()[0]