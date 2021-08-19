from _typeshed import Self
import numpy as np
import random
from numpy.random.mtrand import permutation
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold

class explain(object):
    
    def __init__(self, model, poi):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[1]))
        self.fra = np.array([])
        
    def feasible_recourse_actions(self, data, k=100, certainty=0.7, bandwidth= None, density=0.7):
        sorted_closest_points = np.array(sorted([(np.linalg.norm(data[i] - self.poi[0]), data[i]) for i in range(data.shape[0])], key = lambda row: row[0]), dtype=object)[:, 1]
        fra = list()
        kde = GridSearchCV(KDE(), {'bandwidth': np.logspace(-1, 1, 20)}, cv=KFold(n_splits = 5), n_jobs=-1).fit(data).best_estimator_ if bandwidth == None else KDE(bandwidth=bandwidth).fit(data)
        max_density = max(kde.score_samples(data))
        for point in sorted_closest_points:
            if self.model.predict(self.poi) != (clas := self.model.predict([point])) and self.model.predict_proba([point])[0][clas] >= certainty and np.exp(kde.score_samples([point])[0]) >= (density * max_density):
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
    
    def _minimal_causes(self, S, i, quasi = False):
        if S[i] != 0:
            temp = S.copy()
            temp[i] = 0
            if (vos := self.value(S)) == 1 and vos != self.value(temp):
                if not quasi:
                    temp[i] = 1
                    for j in range(len(S)):
                        if S[j] != 0 and j != i:
                            temp[j] = 0
                            if (vos := self.value(S)) == 1 and vos != self.value(temp):
                                if quasi:
                                    return True
                                else:
                                    temp[j] = 1
                            else:
                                if quasi:
                                    temp[j] = 1
                                else:
                                    return False
                return True
        return False

    def sample(self, δ, ε, index_type, seed=0):
        indices = { "Johnston": lambda S, i: (1 / len(self._critical_features(S))) if self._minimal_causes(S, i, quasi = True) else 0, 
                    "Deegan-Packel": lambda S, i: (1 / np.sum(S)) if self._minimal_causes(S, i) else 0, 
                    "Holler-Packel": lambda S, i: 1 if self._minimal_causes(S, i) else 0}
        if index_type not in indices:
            raise ValueError("Can't compute " + index_type + " index")
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(np.log(2 * len(self.N) / δ) / (2 * np.power(ε, 2))))
        random.seed(seed)            
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            unbiased_estimate = np.array(list(map(lambda i: unbiased_estimate[i] + (2 * indices[index_type](S, i)) , np.arange(len(self.N)))))
        return unbiased_estimate / samples

    def responsibility_index(self, δ, ε, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil((np.log(1 / ε) + np.log(len(self.N) / δ)) / ε))
        random.seed(seed)            
        for _ in range(samples):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            for i in self.N:
                if self._minimal_causes(S, i, quasi = True):
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1 / np.sum(S))
        return unbiased_estimate
    
    def banzhaf_index(self, δ, ε, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(len(self.N) * np.log(len(self.N) / δ) / np.power(ε, 2)))
        random.seed(seed)
        for _ in range(samples):
            for i in self.N:
                coalition = [random.randint(0, 1) if i != j else 1 for j in range(len(self.N))]
                unbiased_estimate[i] += 1 if self._minimal_causes(coalition, i, quasi=True) else 0
        return unbiased_estimate / samples

    def shapley_shubik_index(self, δ, ε, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        samples = int(np.ceil(len(self.N) * np.log(len(self.N) / δ) / np.power(ε, 2)))
        random.seed(seed)
        for _ in range(samples):
            permutation = np.random.permutation(len(self.N))
            for i in self.N:
                S = np.zeros(len(self.N))
                S[permutation[: i + 1]] = 1
                if self._minimal_causes(S, i): 
                    unbiased_estimate[i] += 1 
                    break
        return unbiased_estimate / samples