import numpy as np
from itertools import chain, combinations
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.ensemble import RandomForestClassifier
from Data import get_German_Data, get_Adult_Data
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt 

class explain(object):
    
    def __init__(self, model, poi):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[1]))
        self.fra = np.array([])
        self.value_cache = dict()
        self.critical_features_cache = dict()
        self.minimality_cache = dict()
        
    def feasible_recourse_actions(self, data, out, k=100, certainty=0.7, bandwidth=None, density=0.7):
        sorted_closest_points = np.array(sorted([(np.linalg.norm(data[i] - self.poi[0]), data[i], out[i]) for i in range(data.shape[0])], key = lambda row: row[0]), dtype=object)[:, 1:]
        fra = list()
        kde = GridSearchCV(KDE(), {'bandwidth': np.logspace(-1, 1, 20)}, cv=KFold(n_splits = 5), n_jobs=-1).fit(data).best_estimator_ if bandwidth == None else KDE(bandwidth=bandwidth).fit(data)
        max_density = max(kde.score_samples(data))
        for point, actual in sorted_closest_points:
            if self.model.predict(self.poi) != (clas := self.model.predict([point])) and clas == actual and self.model.predict_proba([point])[0][clas] >= certainty and np.exp(kde.score_samples([point])[0]) >= (density * max_density):
                fra.append(point)
                k -= 1
                if k == 0:
                    break
        self.fra = np.array(fra)
        return self

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
    
    def Responsiblity_index(self):
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
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] += 2 
        return unbiased_estimate / num_samples
    
    def Banzhaf_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] += 1
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Shapley_Shubik_sample(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        np.random.seed(seed)
        samples = np.random.randint(0, 2, (num_samples, len(self.N)))
        for S in samples:
            size_S = np.sum(S)
            n = len(self.N)
            addend = np.power(2, n) * np.math.factorial(n - size_S) * np.math.factorial(size_S - 1) / np.math.factorial(n)  
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] +=  addend
        return unbiased_estimate / num_samples
    
    def Shapley_Shubik_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            size_S = np.sum(S)
            n = len(self.N)
            addend = np.math.factorial(size_S - 1) * np.factorial(n - size_S) / np.math.factorial(n)
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] += addend
        return unbiased_estimate

def main():
    X_trn, X_tst, Y_trn, Y_tst = get_Adult_Data()
    model = RandomForestClassifier(n_estimators=50, max_features=None, n_jobs=-1, random_state=0).fit(X_trn, Y_trn)
    poi0 = np.array([X_tst[0]])
    poi1 = np.array([X_tst[3]])
    print(model.predict(poi0))
    print(model.predict(poi1))
    data = np.delete(X_tst[1:], 3, 0)
    out = np.delete(Y_tst[1:], 3, 0)
    ε = 1e-2
    δ = 1e-4
    print("poi0")
    exp = explain(model, poi0).feasible_recourse_actions(data, out, 5, 0.7, bandwidth=None, density=0.7)
    print("J:")
    print(exp.Johnston_sample(ε, δ))
    print("D:")
    print(exp.Deegan_Packel_sample(ε, δ))
    print("H:")
    print(exp.Holler_Packel_sample(ε, δ))
    print("R:")
    print(exp.Responsibility_sample(ε, δ))
    print("B:")
    print(exp.Banzhaf_sample(ε, δ))
    print("S:")
    print(exp.Shapley_Shubik_sample(ε, δ))
    print("poi1")
    exp = explain(model, poi1).feasible_recourse_actions(data, out, 5, 0.7, bandwidth=None, density=0.7)
    print("J:")
    print(exp.Johnston_sample(ε, δ))
    print("D:")
    print(exp.Deegan_Packel_sample(ε, δ))
    print("H:")
    print(exp.Holler_Packel_sample(ε, δ))
    print("R:")
    print(exp.Responsibility_sample(ε, δ))
    print("B:")
    print(exp.Banzhaf_sample(ε, δ))
    print("S:")
    print(exp.Shapley_Shubik_sample(ε, δ))


if __name__ == "__main__":
    main()