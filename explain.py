from operator import add
import numpy as np
from itertools import chain, combinations
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.ensemble import RandomForestClassifier
from data import get_German_Data, get_Adult_Data, get_ACS_Data
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

def Node_Models(X, nodes, edges, seed=0):
    node_f = dict()
    sources = set()
    for node in nodes:
        if node not in set([edge[1] for edge in edges]):
            sources.add(node)
    
    def get_parents(node):
        parents = set()
        for edge in edges:
            if edge[1] == node:
                parents.add(edge[0])
        return parents

    for node in nodes:
        if node not in sources:
            parents = get_parents(node)
            X_train, X_test, y_train, y_test = train_test_split(X[list(parents)], np.array(X[node]), test_size=0.2, random_state=seed)
            X_train = X_train.rename(columns={key: value for value, key in enumerate(X_train.columns)})
            X_test = X_test.rename(columns={key: value for value, key in enumerate(X_test.columns)})
            categorical = set(X_train.columns) - set(X_train._get_numeric_data().columns)
            X_train = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [feature for feature in X_train.columns if feature in categorical])], remainder='passthrough', n_jobs=-1).fit_transform(X_train)
            X_test = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [feature for feature in X_test.columns if feature in categorical])], remainder='passthrough', n_jobs=-1).fit_transform(X_test)
            y_train = LabelEncoder().fit_transform(np.array(y_train))
            y_test = LabelEncoder().fit_transform(np.array(y_test))
            xgb_train = xgboost.DMatrix(X_train, label=y_train)
            xgb_test = xgboost.DMatrix(X_test, label=y_test)
            if node in categorical:
                num_class = len(np.unique(X[node]))
                params = {
                    "eta": 0.002,
                    "max_depth": 3,
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'num_class': num_class,
                    "subsample": 0.5
                }
            else:
                params = {
                "eta": 0.002,
                "max_depth": 3,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                "subsample": 0.5
            }
            node_f[node] = xgboost.train(params, xgb_train, 500, evals = [(xgb_test, "test")], verbose_eval=100)
    for node in sources:
        node_f[node] = None
    return node_f

class explain(object):
    
    def __init__(self, model, poi, node_f):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[1]))
        self.fra = np.array([])
        self.value_cache = dict()
        self.critical_features_cache = dict()
        self.minimality_cache = dict()
        self.node_f = node_f
        
    def feasible_recourse_actions(self, data, out, k=100, certainty=0.7, bandwidth=None, density=0.7):
        sorted_closest_points = np.array(sorted([(np.linalg.norm(data[i] - self.poi[0]), data[i], out[i]) for i in range(data.shape[0])], key = lambda row: row[0]), dtype=object)[:, 1:]
        fra = list()
        kde = GridSearchCV(KDE(), {'bandwidth': np.logspace(-1, 1, 20)}, cv=KFold(n_splits = 5), n_jobs=-1).fit(data).best_estimator_ if bandwidth == None else KDE(bandwidth=bandwidth).fit(data)
        density_thresh = density * max(kde.score_samples(data))
        for point, actual in sorted_closest_points:
            if self.model.predict(self.poi) != (clas := self.model.predict([point])) and clas == actual and self.model.predict_proba([point])[0][clas] >= certainty and np.exp(kde.score_samples([point])[0]) >= density_thresh:
                fra.append(point)
                k -= 1
                if k == 0:
                    break
        self.fra = np.array(fra)
        return self
    

    def _fit_missing_nodes(self, changed_nodes):
        pass

    def value(self, S):
        if len(self.fra) == 0:
            raise ValueError("There are no feasible recourse actions")
        if (str_S := np.array2string(S, separator='')[1:-1]) in self.value_cache:
            return self.value_cache[str_S]
        for point in self.fra:
            xp = ((point * S) + (self.poi[0] * (1 - S))).reshape(1, -1)
            if self.model.predict(xp) != self.model.predict(self.poi):
                self.value_cache[str_S] = 1
                return 1
            self.value_cache[str_S] = 0
        return 0
     
    def _critical_features(self, S):
        if (str_S := np.array2string(S, separator='')[1:-1]) in self.critical_features_cache:
            return self.critical_features_cache[str_S]
        χ = np.zeros(len(self.N), dtype=int)
        for i in range(len(S)):
            if S[i] != 0:
                vos = self.value(S)
                S[i] = 0
                if vos == 1 and self.value(S) == 0:
                    χ[i] = 1
                S[i] = 1
        self.critical_features_cache[str_S] = χ
        return χ
    
    def _is_quasi_minimal(self, S, i):
        if S[i] != 0:
            vos = self.value(S)
            S[i] = 0
            if vos == 1 and self.value(S) == 0:
                S[i] = 1
                return True
            S[i] = 1
        return False
    
    def _is_minimal(self, S):
        if (str_S := np.array2string(S, separator='')[1:-1]) in self.minimality_cache:
            return self.minimality_cache[str_S]
        for i in range(len(S)):
            if S[i] != 0:
                vos = self.value(S)
                S[i] = 0
                if vos == 1 and self.value(S) == 0:
                    S[i] = 1
                else:
                    S[i] = 1
                    self.minimality_cache[str_S] = False
                    return False
        self.minimality_cache[str_S] = True
        return True
    
    def Johnston_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        np.random.seed(seed)
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        for S in samples:
            χ = self._critical_features(S)
            if (size_χ := np.sum(χ)) != 0: 
                unbiased_estimate += (2 * χ / size_χ)
        return unbiased_estimate / num_samples
    
    def Johnston_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            χ = self._critical_features(S)
            if (size_χ := np.sum(χ)) != 0: 
                unbiased_estimate += (χ / size_χ)
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Deegan_Packel_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        np.random.seed(seed)        
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        for S in samples:
            if self._is_minimal(S) and (size_S := np.sum(S)) != 0:
                unbiased_estimate += (2 * S / size_S)
        return unbiased_estimate / num_samples
    
    def Deegan_Packel_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            if self._is_minimal(S) and (size_S := np.sum(S)) != 0:
                unbiased_estimate += (S / size_S)
        return unbiased_estimate / np.power(2, len(self.N) - 1)
    
    def Holler_Packel_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        np.random.seed(seed)        
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        for S in samples:
            if self._is_minimal(S):
                unbiased_estimate += 2 * S
        return unbiased_estimate / num_samples
    
    def Holler_Packel_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            if self._is_minimal(S):
                unbiased_estimate += S
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Responsibility_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil((np.log(1 / ε) + np.log(len(self.N) / δ)) / ε))
        np.random.seed(seed)            
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        for S in samples:
            size_S = np.sum(S)
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1 / size_S)
        return unbiased_estimate
    
    def Responsibility_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            if self._is_minimal(S):
                size_S = np.sum(S)
                for i in self.N:
                    if S[i] == 1:
                        unbiased_estimate[i] = max(unbiased_estimate[i], 1 / size_S)
        return unbiased_estimate 
    
    def Banzhaf_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        np.random.seed(seed)
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        for S in samples:
            unbiased_estimate += np.vectorize(lambda i, S=S: 2 if self._is_quasi_minimal(S, i) else 0)(np.arange(len(self.N)))
        return unbiased_estimate / num_samples
    
    def Banzhaf_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            unbiased_estimate += np.vectorize(lambda i, S=S: 1 if self._is_quasi_minimal(S, i) else 0)(np.arange(len(self.N)))
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Shapley_Shubik_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        np.random.seed(seed)
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        n = len(self.N)
        n_fact = np.math.factorial(n)
        two_to_n = np.power(2., n)
        for S in samples:
            if (size_S := np.sum(S)) != 0:
                addend = two_to_n * np.math.factorial(size_S - 1) * np.math.factorial(n - size_S) / n_fact
                for i in self.N:
                    if self._is_quasi_minimal(S, i):
                        unbiased_estimate[i] += addend 
        return unbiased_estimate / num_samples
    
    def Shapley_Shubik_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        n = len(self.N)
        n_fact = np.math.factorial(n)
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            if (size_S := int(np.sum(S))) != 0:
                addend = np.math.factorial(size_S - 1) * np.math.factorial(n - size_S) / n_fact
                for i in self.N:
                    if self._is_quasi_minimal(S, i):
                        unbiased_estimate[i] += addend
        return unbiased_estimate

def main():
    train = pd.read_csv("adult.data", header=None, na_values= ' ?')
    train = train.dropna()
    train.drop([2, 3, 13, 14], axis=1, inplace=True)
    train = train.rename(columns={key: value for value, key in enumerate(train.columns)})
    node_f = Node_Models(train, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {(0, 2), (0,3), (0, 4), (1, 5), (1, 3), (1, 6), (3, 7), (6, 8), (5, 9), (6, 10)})
    print(node_f)


if __name__ == "__main__":
    main()