import numpy as np
import random
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold

class explain(object):
    
    def __init__(self, model, poi):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[1]))
        self.fra = np.array([])
        
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
        for point in self.fra:
            xp = list()
            for i in range(point.shape[0]):
                if S[i] == 1: 
                    xp.append(point[i])
                else:
                    xp.append(self.poi[0][i])
            if self.model.predict([xp]) != self.model.predict(self.poi):
                return 1
        return 0
    
    def _critical_features(self, S):
        χ = set()
        for i in range(len(S)):
            if S[i] == 1:
                temp = S.copy()
                temp[i] = 0
                if (vos := self.value(S)) == 1 and vos != self.value(temp):
                    χ.add(i)
        return χ
    
    def _is_quasi_minimal(self, S, i):
        if S[i] != 0:
            temp = S.copy()
            temp[i] = 0
            if (vos := self.value(S)) == 1 and vos != self.value(temp):
                return True
        return False

    def _is_minimal(self, S):
        temp = S.copy()
        for i in range(len(S)):
            if S[i] != 0 :
                temp[i] = 0
                if (vos := self.value(S)) == 1 and vos != self.value(temp):
                    temp[i] = 1
                else:
                    return False
        return True
    
    def Johnston_index(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        random.seed(seed)        
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            size_χ = len(self._critical_features(S))
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] += (2 / size_χ)
        return unbiased_estimate / samples
    
    def Deegan_Packel_index(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        random.seed(seed)        
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            if self._is_minimal(S):
                size_S = np.sum(S)
                for i in self.N:
                    if self._is_quasi_minimal(S, i):
                        unbiased_estimate[i] += (2 / size_S)
        return unbiased_estimate / samples
    
    def Holler_Packel_index(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        random.seed(seed)        
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            if self._is_minimal(S):
                for i in self.N:
                    if self._is_quasi_minimal(S, i):
                        unbiased_estimate[i] += 2
        return unbiased_estimate / samples

    def Responsibility_index(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil((np.log(1 / ε) + np.log(len(self.N) / δ)) / ε))
        random.seed(seed)            
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            size_S = np.sum(S)
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1 / size_S)
        return unbiased_estimate
    
    def Banzhaf_index(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        random.seed(seed)
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] += 2 
        return unbiased_estimate / samples

    def Shapley_index(self, ε, δ, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        random.seed(seed)
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            size_S = np.sum(S)
            n = len(self.N)
            addend = np.power(2, n) * np.math.factorial(n - size_S) * np.math.factorial(size_S - 1) / np.math.factorial(n)  
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] +=  addend
        return unbiased_estimate / samples