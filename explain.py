import numpy as np
from itertools import chain, combinations
import math
import multiprocessing as mp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

class Mapper:
    '''
    A class that maps categorical values to either one hot encoded values (ohe) or label encoded values (le)
    '''
    def __init__(self, data):
        self.Data = data
        self.columns = sorted(data.columns)
        self.categorical = list(set(data.columns) - set(data._get_numeric_data()))
        self.mappings = dict()
        self.newData = pd.DataFrame(np.zeros(len(data)), columns=['temp'])
        self.onehot = dict()
        self.label = dict()
        for col in self.columns:
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
        '''
        This function returns the tranformed dataset
        
        Input:
            - X: The columns needed; needs a list of values
            - Y: The column that will act as Y values; needs the column index of the Y value
            - N: Number of points in the data set; an int between 0 to size of dataset
        '''
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
        '''
        Tranforms a categorical value to its ohe value
        '''
        for value in self.mappings[column]:
            if value[0] == label:
                return value[1]

    def le_transform(self, label, column):
        '''
        Tranforms a categorical value to its le value
        '''
        for value in self.mappings[column]:
            if value[0] == label:
                return value[2]
    
    def point_transform(self, vector, X=None, Y=None):
        '''
        Tranforms a whole point to ohe values for X columns and le values for the Y column
        Input:
            - vector: vector intended to be transformed
            - X: The columns needed; needs a list of values
            - Y: The column that will act as Y values; needs the column index of the Y value
        '''        
        newPoint = []
        idx = 0
        for col in self.columns[: -1]:
            if X is None or col in X:
                if col in self.categorical:
                    newPoint += list(self.ohe_transform(vector[idx], col))
                else:
                    newPoint += [vector[idx]]
                idx += 1
        if Y != None:
            newPoint += [self.le_transform(vector[idx], Y)]
        return np.array(np.array(newPoint, dtype=float), dtype=int)
    
    def ohe_inverse(self, vector, column):
        '''
        Inverses a ohe value to its categorical value
        '''
        for value in self.mappings[column]:
            if np.array_equal(value[1], vector):
                return value[0]
    
    def le_inverse(self, label, column):
        '''
        Inverses a le value to its categorical value
        '''
        for value in self.mappings[column]:
            if value[2] == label:
                return value[0]
    
    def point_inverse(self, vector, y_present=False):
        '''
        Inverses a whole point to its original self
        '''
        newPoint = []
        offset = 0 
        for col in self.columns[: -1]:
            if col in self.categorical:
                col_len = len(self.mappings[col][0][1])
                newPoint += [self.ohe_inverse(vector[col + offset: col + offset +  col_len], col)]
                offset += col_len - 1
            else:
                newPoint += [vector[col + offset]]
        if y_present:
            newPoint += [self.le_inverse(vector[-1], col + 1 )]
        return np.array(newPoint)


class Causal_Explanations(object):
    '''
    A class for causal model explanations
    '''
    def __init__(self, model, poi, mapper, baselines):
        '''
        Input:
            - model: A black-box model
            - poi: A point of interest whose outcome needs explaining 
            - mapper: An object of type Mapper that converts between actual values to ohe and le values
            - baselines: A set of feasible actions that can be taken to change the outcome of the poi
        '''
        self.model = model
        self.poi = poi
        self.N = set(np.arange(poi.shape[0]))
        self.value_cache = dict()
        self.critical_features_cache = dict()
        self.minimality_cache = dict()
        self.mapper = mapper
        self.poiohe = self.mapper.point_transform(self.poi)
        self.baselines = baselines
    
    def value(self, S):
        '''
        Computes the power of the set S
        '''
        if (str_S := np.array2string(S, separator='')[1:-1]) in self.value_cache:
            return self.value_cache[str_S]
        for point in self.baselines:
            xp = self.mapper.point_transform(np.where(S == 1, point, self.poi)).reshape(1, -1)
            if self.model.predict(xp) != self.model.predict(self.poiohe.reshape(1, -1)):
                self.value_cache[str_S] = 1
                return 1
            self.value_cache[str_S] = 0
        return 0

    def _critical_features(self, S):
        '''
        Returns the set of critical features in S
        '''
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
        vos = self.value(S)
        if(vos == 0):
            return False
        for i in range(len(S)):
            if S[i] != 0:
                S[i] = 0
                if self.value(S) == 0:
                    S[i] = 1
                else:
                    S[i] = 1
                    self.minimality_cache[str_S] = False
                    return False
        self.minimality_cache[str_S] = True
        return True

    def Johnston_sample_helper(self, inp):
        num_samples = inp[0]
        np.random.seed(inp[1])
        n = len(self.N)
        unbiased_estimate = np.zeros(n)
        for m in range(num_samples):
            S = np.random.randint(0, 2, n)
            χ = self._critical_features(S)
            if (size_χ := np.sum(χ)) != 0: 
                unbiased_estimate += (2 * χ / size_χ)
        return unbiased_estimate / num_samples

    def Johnston_sample(self, ε, δ, seed = 0, num_processes = 1):
        '''
        Sampling Algorithm for the Johnston Index
        '''
        num_samples = int(np.ceil(2 * np.log(2 * len(self.N) / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Johnston_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Johnston_index(self):
        '''
        Actual Algorithm for the Johnston Index
        '''
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
            S = np.random.randint(0, 2, n)
            if self._is_minimal(S) and (size_S := np.sum(S)) != 0:
                unbiased_estimate +=  (2 * S / size_S)
        return unbiased_estimate / num_samples

    def Deegan_Packel_sample(self, ε, δ, seed=0, num_processes = 1):
        '''
        Sampling Algorithm for the Deegan Packel Index
        '''
        num_samples = int(np.ceil(2 * np.log(2 * len(self.N) / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Deegan_Packel_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Deegan_Packel_index(self):
        '''
        Actual Algorithm for the Deegan Packel Index
        '''
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
            S = np.random.randint(0, 2, n)
            if self._is_minimal(S):
                unbiased_estimate += 2 * S
        return unbiased_estimate / num_samples
    
    def Holler_Packel_sample(self, ε, δ, seed=0, num_processes = 1):
        '''
        Sampling Algorithm for the Holler Packel Index
        '''
        num_samples = int(np.ceil(2 * np.log(2 * len(self.N) / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Holler_Packel_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Holler_Packel_index(self):
        '''
        Actual Algorithm for the Holler Packel Index
        '''
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
            S = np.random.randint(0, 2, n)
            size_S = np.sum(S)
            for i in self.N:
                if self._is_quasi_minimal(S, i):
                    unbiased_estimate[i] = max(unbiased_estimate[i], 1 / size_S)
        return unbiased_estimate

    def Responsibility_sample(self, ε, δ, seed=0, num_processes = 1):
        '''
        Sampling Algorithm for the Responsibility Index
        '''
        num_samples = int(np.ceil((np.log(1 / ε) + np.log(len(self.N) / δ)) / ε))
        np.random.seed(seed)            
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Responsibility_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.max(result, axis = 0)
    
    def Responsibility_index(self):
        '''
        Actual Algorithm for the Responsibility Index
        '''
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
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k])
            χ = self._critical_features(S)
            unbiased_estimate += (n*math.comb(n, k)/(2**(n-1)))*(χ)
        return unbiased_estimate / num_samples

    def Banzhaf_sample(self, ε, δ, seed = 0, num_processes = 1):
        '''
        Sampling Algorithm for the Banzhaf Index
        '''
        n = len(self.N)
        num_samples = int(2 * n * n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Banzhaf_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Banzhaf_index(self):
        '''
        Actual Algorithm for the Banzhaf Index
        '''
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
            k = np.random.randint(1, n+1)
            X = np.random.permutation(n)
            S = self.to_binary(X[:k])
            χ = self._critical_features(S)
            unbiased_estimate += (n*math.comb(n, k)*(math.factorial(k-1)*math.factorial(n-k))/math.factorial(n))*(χ)
        return unbiased_estimate / num_samples

    def Shapley_Shubik_sample(self, ε, δ, seed = 0, num_processes = 1):
        '''
        Sampling Algorithm for the Shapley Shubik Index
        '''
        n = len(self.N)
        num_samples = int(2 * n * n * np.ceil(np.log(2 * n / δ) / (np.power(ε, 2))))
        np.random.seed(seed)
        with mp.Pool(processes = num_processes) as pool:
            result = pool.map(self.Shapley_Shubik_sample_helper, [[int(np.ceil(num_samples/num_processes)), seed+i] for i in range(num_processes)])
        return np.mean(result, axis = 0)
    
    def Shapley_Shubik_index(self):
        '''
        Actual Algorithm for the Shapley Shubik Index
        '''
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
