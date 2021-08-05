import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from data import getData

class explain(object):
    def __init__(self, model, poi):
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[1]))
        self.fra = np.array([])
        
    def feasible_recourse_actions(self, data, k):
        sorted_closest_points = np.array(sorted([(np.linalg.norm((data[i] - self.poi)), data[i]) for i in range(data.shape[0])], key = lambda row: row[0]))[:, 1]
        fra = list()
        for point in sorted_closest_points:
            if self.model.predict(self.poi) != self.model.predict(point):
                fra.append(point.toarray())
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
                temp[i] = 0
                if self.value(S) != self.value(temp):
                    χ.add(i)
        return χ
    
    def _minimal_causes(self, S, i, quasi = False):
        if S[i] != 0:
            temp = S.copy()
            temp[i] = 0
            if self.value(S) != self.value(temp):
                if not quasi:
                    temp[i] = 1
                    for j in range(len(S)):
                        if S[j] != 0 and j != i:
                            temp[j] = 0
                            if self.value(S) != self.value(temp):
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

    def sample(self, m, index, seed=0):
        indices = { "Johnston": lambda S, i: (1 / len(self._critical_features(S))) if self._minimal_causes(S, i, quasi = True) else 0, 
                    "Deegan-Packel": lambda S, i: (1 / np.sum(S)) if self._minimal_causes(S, i) else 0, 
                    "Holler-Packel": lambda S, i: 1 if self._minimal_causes(S, i) else 0}
        if index not in indices:
            raise ValueError("Can't compute " + index + " index")
        unbiased_estimate = np.zeros(len(self.N))
        random.seed(seed)            
        for _ in range(m):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            unbiased_estimate = np.array(list(map(lambda i: unbiased_estimate[i] + (2 * indices[index](S, i)) , np.arange(len(self.N)))))
        return unbiased_estimate / m

    def responsibility_index(self, m, seed=0):
        unbiased_estimate = np.zeros(len(self.N))
        random.seed(seed)            
        for _ in range(m):
            S = [random.randint(0, 1) for _ in range(len(self.N))]
            for i in self.N:
                if self._minimal_causes(S, i, quasi = True):
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1 / np.sum(S))
        return unbiased_estimate
        
def main():
    X_trn, Y_trn, X_tst, Y_tst = getData()
    model = RandomForestClassifier(n_estimators=50, max_features=None, n_jobs=-1, random_state=0).fit(X_trn, Y_trn)
    poi = X_tst[0].toarray().reshape(1,-1)
    data = X_tst[1: 1000]
    exp = explain(model, poi).feasible_recourse_actions(data, 10)


if __name__ == "__main__":
    main()