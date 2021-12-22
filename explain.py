from operator import add
from joblib.logger import short_format_time
import numpy as np
from itertools import chain, combinations
from numpy.random import pareto
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold
import math
import multiprocessing as mp

from sklearn.ensemble import RandomForestClassifier
from data import get_German_Data, get_Adult_Data, get_ACS_Data, Representer
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

class Casual_Graph():
    def __init__(self, nodes, edges, mapper):
        self.nodes = nodes
        self.edges = edges
        self.mapper = mapper
        self.node_f = dict()
        self.sources = set()
        for node in self.nodes:
            if node not in set([edge[1] for edge in self.edges]):
                self.sources.add(node)
        for node in nodes:
            if node not in self.sources:
                parents = self.get_parents(node)
                data = self.mapper.get_data(parents, node)
                X_train, X_test, Y_train, Y_test = train_test_split(data[: , : -1], data[: , -1], test_size=0.33, shuffle=False)
                xgb_train = xgboost.DMatrix(X_train, label=Y_train)
                xgb_test = xgboost.DMatrix(X_test, label=Y_test)
                if node in self.mapper.categorical:
                    num_class = len(np.unique(data[:, -1]))
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
                self.node_f[node] = xgboost.train(params, xgb_train, 500, evals = [(xgb_test, "test")], verbose_eval=100)
    
    def get_parents(self, node):
        parents = set()
        for edge in self.edges:
            if edge[1] == node:
                parents.add(edge[0])
        return parents
    
    def predict(self, node, point):
        prediction = self.node_f[node].predict(xgboost.DMatrix(point.reshape(1, -1)))[0]
        return np.argmax(prediction) if node in self.mapper.categorical else prediction

class explain(object):
    
    def __init__(self, model, poi, casual_graph, mapper):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[0]))
        self.fra = np.array([])
        self.value_cache = dict()
        self.critical_features_cache = dict()
        self.minimality_cache = dict()
        self.cg = casual_graph
        self.mapper = mapper
        
    def feasible_recourse_actions(self, data, out, k=100, certainty=0.7, bandwidth=None, density=0.7):
        self.poiohe = self.mapper.point_transform(self.poi)
        sorted_closest_points = np.array(sorted([(np.linalg.norm(data[i] - self.poiohe), data[i], out[i]) for i in range(data.shape[0])], key = lambda row: row[0]), dtype=object)[:, 1:]
        fra = list()
        kde = GridSearchCV(KDE(), {'bandwidth': np.logspace(-1, 1, 20)}, cv=KFold(n_splits = 5), n_jobs=-1).fit(data).best_estimator_ if bandwidth == None else KDE(bandwidth=bandwidth).fit(data)
        density_thresh = density * max(kde.score_samples(data))
        for point, actual in sorted_closest_points:
            if self.model.predict(self.poiohe.reshape(1, -1)) != (clas := self.model.predict([point])) and clas == actual and self.model.predict_proba([point])[0][clas] >= certainty and np.exp(kde.score_samples([point])[0]) >= density_thresh:
                fra.append(point)
                k -= 1
                if k == 0:
                    break
        self.fra = np.array(fra)
        return self
    
    def _get_xp(self, S, point):
        newPoint = []
        point = self.mapper.point_inverse(point)
        changed = set()
        for idx, val in enumerate(S):
            if val == 1:
                newPoint.append(point[idx])
                changed.add(idx)
            else:
                newPoint.append(self.poi[idx])
        for idx, val in enumerate(S):
            if val == 0:
                parents = self.cg.get_parents(idx)
                if bool(parents & changed) and idx not in changed:
                    parents = sorted(parents)
                    prediction = self.cg.predict(idx, self.mapper.point_transform(point[parents], parents))
                    newPoint[idx] = self.mapper.le_inverse(prediction, idx) if idx in self.mapper.categorical else prediction
        return self.mapper.point_transform(np.array(newPoint))

    # def value(self, S):
    #     if len(self.fra) == 0:
    #         raise ValueError("There are no feasible recourse actions")
    #     if (str_S := np.array2string(S, separator='')[1:-1]) in self.value_cache:
    #         return self.value_cache[str_S]
    #     for point in self.fra:
    #         xp = self._get_xp(S, point).reshape(1, -1)
    #         if self.model.predict(xp) != self.model.predict(self.poiohe.reshape(1, -1)):
    #             self.value_cache[str_S] = 1
    #             return 1
    #         self.value_cache[str_S] = 0
    #     return 0
     
    def value(self, S):
        if(S[0] == 1):
            return 1
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

    def find_minimal_set(self, S, i):
        n = len(self.N)
        Sret = np.copy(S)
        cant_remove = [i]
        while(1):
            flag1 = 0
            removed = -1
            Sprime = np.copy(Sret)
            for j in range(n):
                if(j in cant_remove):
                    continue
                if(Sret[j] == 1):
                    Sprime[j] == 0
                    removed = j
                    flag1 = 1
                    break
            if(flag1 == 1):
                if(self._is_quasi_minimal(Sprime, i)):
                    Sret = np.copy(Sprime)
                else:
                    cant_remove.append(removed)
            if(len(cant_remove) == np.sum(Sret)):
                break
            if(flag1 == 0):
                break
        return Sret

    def to_binary(self, X, n):
        Y = np.array([0]*n)
        for x in X:
            Y[x] = 1
        return Y 

    def Johnston_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            # print(m)
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k], n)
            χ = self._critical_features(S)
            if (size_χ := np.sum(χ)) != 0: 
                unbiased_estimate += (n*math.comb(n, k)/(2**(n-1)))*(χ / size_χ)
        return unbiased_estimate / num_samples

    def Johnston_sample(self, ε, δ, seed = 0, num_processes = 1):
        # print("N = %d" % len(self.N))
        n = len(self.N)
        num_samples = int(n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Johnston_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
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

    def Deegan_Packel_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k], n)
            if self._is_minimal(S) and (size_S := np.sum(S)) != 0:
                unbiased_estimate += (n*math.comb(n, k)/(2**(n-1)))*(S / size_S)
        return unbiased_estimate / num_samples

    def Deegan_Packel_sample(self, ε, δ, seed=0, num_processes = 1):
        n = len(self.N)
        num_samples = int(n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Deegan_Packel_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Deegan_Packel_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            if self._is_minimal(S) and (size_S := np.sum(S)) != 0:
                unbiased_estimate += (S / size_S)
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Holler_Packel_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k], n)
            if self._is_minimal(S):
                unbiased_estimate += (n*math.comb(n, k)/(2**(n-1)))*(S)
        return unbiased_estimate / num_samples
    
    def Holler_Packel_sample(self, ε, δ, seed=0, num_processes = 1):
        n = len(self.N)
        num_samples = int(n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Holler_Packel_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Holler_Packel_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            if self._is_minimal(S):
                unbiased_estimate += S
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Responsibility_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k], n)
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    # size_S = np.sum(self.find_minimal_set(S, i))
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1 / np.sum(S))
        return unbiased_estimate


    def Responsibility_sample(self, ε, δ, seed=0, num_processes = 1):
        n = len(self.N)
        num_samples = int(np.ceil((np.log(1 / ε) + np.log(len(self.N) / δ)) / ε))
        np.random.seed(seed)            
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Responsibility_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.max(result, axis = 0)
    
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

    def Banzhaf_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            # print(m)
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k], n)
            χ = self._critical_features(S)
            unbiased_estimate += (n*math.comb(n, k)/(2**(n-1)))*(χ)
        return unbiased_estimate / num_samples

    def Banzhaf_sample(self, ε, δ, seed = 0, num_processes = 1):
        # print("N = %d" % len(self.N))
        n = len(self.N)
        num_samples = int(n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Banzhaf_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Banzhaf_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            unbiased_estimate += np.vectorize(lambda i, S=S: 1 if self._is_quasi_minimal(S, i) else 0)(np.arange(len(self.N)))
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Shapley_Shubik_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            # print(m)
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k], n)
            χ = self._critical_features(S)
            unbiased_estimate += (n*math.comb(n, k)*(math.factorial(k-1)*math.factorial(n-k))/math.factorial(n))*(χ)
        return unbiased_estimate / num_samples

    def Shapley_Shubik_sample(self, ε, δ, seed = 0, num_processes = 1):
        # print("N = %d" % len(self.N))
        n = len(self.N)
        num_samples = int(n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Shapley_Shubik_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
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
    train = pd.read_csv("data/adult.data", header=None, na_values= ' ?')
    test = pd.read_csv("data/adult.test", header=None, na_values= ' ?') 
    train = train.dropna()
    test = test.dropna()
    train.drop([2, 3, 13], axis=1, inplace=True)
    test.drop([2, 3, 13], axis=1, inplace=True)
    # print(train.loc[0])
    data = pd.concat([train, test], ignore_index=True)
    rep = Representer(data.rename(columns={key: value for value, key in enumerate(data.columns)}))
    cg = Casual_Graph({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {(0, 11), (1, 11), (2, 11), (3, 11), (4, 11), (5, 11), (6, 11), (7, 11), (8, 11), (9, 11), (10, 11)}, rep)
    data = rep.get_data()
    X = data[:, : -1]
    Y = data[:, -1]
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.333, shuffle=False)
    model = RandomForestClassifier(n_estimators=50, max_features=None, n_jobs=-1, random_state=0).fit(X_trn, Y_trn)
    print(model.score(X_trn, Y_trn), model.score(X_tst, Y_tst))
    print(X_tst[0])
    poi = rep.point_inverse(X_tst[0])
    print(poi)
    data = X_tst[1:]
    out = Y_tst[1:]
    # exp = explain(model, poi, cg, rep).feasible_recourse_actions(data, out, 5)
    exp = explain(model, poi, cg, rep)
    print(exp.Johnston_sample(1e-1, 1e-4, num_processes = 2))
    print(exp.Deegan_Packel_sample(1e-1, 1e-4, num_processes = 2))
    print(exp.Holler_Packel_sample(1e-1, 1e-4, num_processes = 2))
    print(exp.Responsibility_sample(1e-1, 1e-4, num_processes = 2))
    print(exp.Banzhaf_sample(1e-1, 1e-4, num_processes = 2))
    print(exp.Shapley_Shubik_sample(1e-1, 1e-4, num_processes = 2))



if __name__ == "__main__":
    main()