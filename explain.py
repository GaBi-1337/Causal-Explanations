from operator import add
import numpy as np
from itertools import chain, combinations
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold


class explain(object):
    
    def __init__(self, model, poi):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[1]))
        self.fra = np.array([])
        self.value_cache = dict()
        self.critical_features_cache = dict()
        self.minimality_cache = dict()
        
    def feasible_recourse_actions(self, data, out, k=10, certainty=0.7, bandwidth=None, density=0.7):
        sorted_closest_points = np.array(sorted([(np.linalg.norm(data[i] - self.poi[0]), data[i], out[i]) for i in range(data.shape[0])], key = lambda row: row[0]), dtype=object)[:, 1:]
        fra = list()
        # kde = GridSearchCV(KDE(), {'bandwidth': np.logspace(-1, 1, 20)}, cv=KFold(n_splits = 5), n_jobs=-1).fit(data).best_estimator_ if bandwidth == None else KDE(bandwidth=bandwidth).fit(data)
        # density_thresh = density * np.exp(max(kde.score_samples(data)))
        density_thresh = 0
        for point, actual in sorted_closest_points:
            clas = self.model.predict([point])
            # if self.model.predict(self.poi) != (clas) and clas == actual and self.model.predict_proba([point])[0][clas] >= certainty and np.exp(kde.score_samples([point])[0]) >= density_thresh:
            if self.model.predict(self.poi) != (clas) and clas == actual and self.model.predict_proba([point])[0][clas] >= certainty:
                fra.append(point)
                k -= 1
                if k == 0:
                    break
        self.fra = np.array(fra)
        return self

    def value(self, S):
        if len(self.fra) == 0:
            raise ValueError("Run feasible recourse actions first")
        str_S = np.array2string(S, separator='')[1:-1]
        if (str_S) in self.value_cache:
            return self.value_cache[str_S]
        for point in self.fra:
            xp = ((point * S) + (self.poi[0] * (1 - S))).reshape(1, -1)
            if self.model.predict(xp) != self.model.predict(self.poi):
                self.value_cache[str_S] = 1
                return 1
            self.value_cache[str_S] = 0
        return 0
     
    def _critical_features(self, S):
        str_S = np.array2string(S, separator='')[1:-1]
        if (str_S) in self.critical_features_cache:
            return self.critical_features_cache[str_S]
        chi = np.zeros(len(self.N), dtype=int)
        for i in range(len(S)):
            if S[i] != 0:
                vos = self.value(S)
                S[i] = 0
                if vos == 1 and self.value(S) == 0:
                    chi[i] = 1
                S[i] = 1
        self.critical_features_cache[str_S] = chi
        return chi
    
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
        str_S = np.array2string(S, separator='')[1:-1]
        if (str_S) in self.minimality_cache:
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
    
    def Johnston_sample(self, eps, delta, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(len(self.N) * np.log(2 * len(self.N) / delta) / (np.power(eps, 2))))
        np.random.seed(seed)
        for j in range(num_samples):
            k = np.random.randint(1, len(self.N) + 1)
            X = np.random.permutation(len(self.N))
            S = np.zeros(len(self.N))
            for kp in range(k):
                S[X[kp]] = 1
            chi = self._critical_features(S)
            size_chi = np.sum(chi)
            if (size_chi) != 0: 
                unbiased_estimate += (2 * chi / size_chi)
        return unbiased_estimate / num_samples
    
    def Johnston_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            chi = self._critical_features(S)
            size_chi = np.sum(chi)
            if (size_chi) != 0: 
                unbiased_estimate += (chi / size_chi)
        return unbiased_estimate / np.power(2, len(self.N) - 1)

    def Deegan_Packel_sample(self, eps, delta, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        # num_samples = int(np.ceil(len(self.N) * np.log(2 * len(self.N) / delta) / (np.power(eps, 2))))
        num_samples = 1024
        np.random.seed(seed)
        for j in range(num_samples):
            print(str(j) + " out of " + str(num_samples), end = '\r')
            k = np.random.randint(1, len(self.N) + 1)
            X = np.random.permutation(len(self.N))
            S = np.zeros(len(self.N))
            for kp in range(k):
                S[X[kp]] = 1
            size_S = np.sum(S)
            if self._is_minimal(S) and (size_S) != 0:
                unbiased_estimate += (2 * S / size_S)
        return unbiased_estimate / num_samples
    
    def Deegan_Packel_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            size_S = np.sum(S)
            if self._is_minimal(S) and (size_S) != 0:
                unbiased_estimate += (S / size_S)
        return unbiased_estimate / np.power(2, len(self.N) - 1)
    
    def Holler_Packel_sample(self, eps, delta, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(len(self.N) * np.log(2 * len(self.N) / delta) / (np.power(eps, 2))))
        np.random.seed(seed)
        for j in range(num_samples):
            k = np.random.randint(1, len(self.N) + 1)
            X = np.random.permutation(len(self.N))
            S = np.zeros(len(self.N))
            for kp in range(k):
                S[X[kp]] = 1
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

    def Responsibility_sample(self, eps, delta, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil((np.log(1 / eps) + np.log(len(self.N) / delta)) / eps))
        np.random.seed(seed)
        for j in range(num_samples):
            k = np.random.randint(1, len(self.N) + 1)
            X = np.random.permutation(len(self.N))
            S = np.zeros(len(self.N))
            for kp in range(k):
                S[X[kp]] = 1
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
    
    def Banzhaf_sample(self, eps, delta, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        num_samples = int(np.ceil(len(self.N) * np.log(2 * len(self.N) / delta) / (np.power(eps, 2))))
        np.random.seed(seed)
        for j in range(num_samples):
            k = np.random.randint(1, len(self.N) + 1)
            X = np.random.permutation(len(self.N))
            S = np.zeros(len(self.N))
            for kp in range(k):
                S[X[kp]] = 1
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

    def Shapley_Shubik_sample(self, eps, delta, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        for i in self.N:
            num_samples = int(np.ceil(np.log(2 * len(self.N) / delta) / (2 * np.power(eps, 2))))
            np.random.seed(seed)
            for j in range(num_samples):
                X = np.random.permutation(len(self.N))
                S = np.zeros(len(self.N))
                for ip in self.N:
                    if(X[ip] == i):
                        S[X[ip]] = 1
                        break
                    S[X[ip]] = 1
                if self._is_quasi_minimal(S, i):
                        unbiased_estimate[i] += 1
        return unbiased_estimate / num_samples
    
    def Shapley_Shubik_index(self):
        unbiased_estimate = np.zeros(len(self.N))
        power_set = set(chain.from_iterable(combinations(self.N, r) for r in range(len(self.N)+1)))
        n = len(self.N)
        n_fact = np.math.factorial(n)
        for subset in power_set:
            S = np.zeros(len(self.N))
            S[list(subset)] = 1
            size_S = int(np.sum(S))
            if size_S != 0:
                addend = np.math.factorial(size_S - 1) * np.math.factorial(n - size_S) / n_fact
                for i in self.N:
                    if self._is_quasi_minimal(S, i):
                        unbiased_estimate[i] += addend
        return unbiased_estimate
