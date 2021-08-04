import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from data import getData

class explain(object):
    def __init__(self, model, poi, data, k=10):
        self.model = model
        self.poi = poi
        self.data = data
        self.N = set(np.arange(poi.shape[0]))
        self.k = k
        
    def feasible_recourse_actions(self):
        sorted_closest_points = np.array(sorted([(np.linalg.norm((self.data[i] - self.poi)), self.data[i]) for i in range(self.data.shape[0])], key = lambda row: row[0]))[:, 1]
        k_nearest_points = list()
        k_counter = 0
        for point in sorted_closest_points:
            if self.model.predict(self.poi) != self.model.predict(point):
                k_nearest_points.append(point.toarray())
                k_counter += 1
                if k_counter == self.k:
                    break
        return np.array(k_nearest_points)

    def value(self, S):
        fra = self.feasible_recourse_actions()
        for point in fra:
            xp = list()
            for i in range(point[0].shape[0]):
                if S[i] == 1: 
                    xp.append(point[0][i])
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
                temp[i] = 0 if temp[i] == 1 else 1
                if self.value(S) != self.value(temp):
                    χ.add(i)
        return χ
    
    def _minimal_causes(self, S, i, quasi = False):
        temp = S.copy()
        temp[i] = 0 if temp[i] == 1 else 1
        if self.value(S) == self.value(temp):
            return False
        for j in range(len(S)):
            if S[j] == 1 and j != i:
                temp[j] = 0 if temp[j] == 1 else 1
                if self.value(S) != self.value(temp):
                    if quasi:
                        return True
                    else:
                        continue
                else:
                    if quasi:
                        continue
                    else:
                        return False
        return True

    def sample(self, m, index, seed=0):
        indices = { "Johnston": lambda S, i: (1 / len(self._critical_features(S))) if self._minimal_causes(S, i, quasi = True) else 0, 
                    "Deegan-Packel": lambda S, i: (1 / S.count(1)) if self._minimal_causes(S, i) else 0, 
                    "Holler-Packel": lambda S, i: 1 if self._minimal_causes(S, i) else 0}
        if index not in indices:
            raise ValueError("Can't compute " + index + " index")
        unbiased_estimate = np.zeros(len(self.N))
        random.seed(seed)            
        for _ in range(m):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            for i in self.N:
                unbiased_estimate[i] += 2 * indices[index](S, i)
        return unbiased_estimate / m

    def responsibility_index(self, m, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        random.seed(seed)            
        for _ in range(m):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            for i in self.N:
                if self._minimal_causes(S, i, quasi = True):
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1/S.count(1))
        return unbiased_estimate