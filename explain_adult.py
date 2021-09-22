import numpy as np 
from data import get_German_Data, get_Adult_Data, get_ACS_Data
from train_models import best_trained_RF
from explain import explain
import pickle
from timeit import default_timer as timer
import torch
import math

from operator import add
import numpy as np
from itertools import chain, combinations
from sklearn.neighbors import KernelDensity as KDE
from sklearn.model_selection import GridSearchCV, KFold
from numba import cuda
from numba import *

eps = 1e-1
delta = 1e-2
flag = 0

def predict(X, w):
	return torch.sigmoid(w[0] + torch.dot(w[1:], X))

def train(X, Y):
	X_trn = torch.tensor(X, dtype = torch.float)
	Y_trn = torch.tensor(Y, dtype = torch.float)
	lr_rate = 1e-3
	epochs = 10
	criterion = torch.nn.BCELoss()
	w = torch.randn(len(X[0]) +1, dtype = torch.float, requires_grad = True)
	optimizer = torch.optim.SGD([w], lr=lr_rate)

	for epoch in range(epochs):
		print(epoch, end = '\r')
		loss = 0
		optimizer.zero_grad()
		for i in range(len(X_trn)):
			y = predict(X_trn[i], w)
			loss += criterion(y, Y_trn[i])
		loss.backward()
		optimizer.step()
		print("Loss = "+str(loss.item()))
	return w.detach().numpy()

def accuracy(gpu_w, X, Y, th):
	count = 0
	for i in range(len(X)):
		y = predict_final(gpu_w, X[i], th)
		if(y == Y[i]):
			count+=1
	return count/len(X)


def predict_final(gpu_w, X, th):
	val = gpu_w[0]
	for i in range(len(X)):
		val += gpu_w[i+1]*X[i]
	if(val < -10):
		return 0
	theta = 1/(1+math.exp(-val))
	if(theta > th):
		return 1
	else:
		return 0

def feasible_recourse_actions(data, out, k=10, poi):
    sorted_closest_points = np.array(sorted([(np.linalg.norm(data[i] - self.poi[0]), data[i], out[i]) for i in range(data.shape[0])], key = lambda row: row[0]), dtype=object)[:, 1:]
    fra = list()
    density_thresh = 0
    for point, actual in sorted_closest_points:
        clas = predict_gpu(point)
        if predict_gpu(poi) != (clas) and clas == actual:
            fra.append(point)
            k -= 1
            if k == 0:
                break
    return np.array(fra)

@cuda.jit(device = True)
def value(fra, S, poi, newpoint, n):
    for k in range(len(fra)):
    	for i in range(n):
    		if(S[i] == 1):
    			newpoint[i] = fra[k, i]
    		else:
    			newpoint[i] = poi[k, i]
        if predict_gpu(newpoint) != predict_gpu(poi):
            return 1
    return 0

@cuda.jit(device = True)
def _is_minimal(S):
    for i in range(len(S)):
        if S[i] != 0:
            vos = value(S)
            S[i] = 0
            if vos == 1 and value(S) == 0:
                S[i] = 1
            else:
                S[i] = 1
                return False
    return True

@cuda.jit(device=True)
def Deegan_Packel_sample(n, estimate, num_samples):
    j = cuda.BlockIdx.x
        
        if self._is_minimal(S) and (size_S) != 0:
            unbiased_estimate += (2 * S / size_S)
    return unbiased_estimate / num_samples


X_trn, X_tst, Y_trn, Y_tst = get_Adult_Data()
pkl_filename = "adult_model.pkl"
try:
	with open(pkl_filename, 'rb') as file:
		model = pickle.load(file)
except:
	flag = 1

if(flag == 1):
	w = train(X_trn, Y_trn)
	print(w)
	print("Training Set Accuracy: ", accuracy(w, X_trn, Y_trn, 0.5))
	print("Test Set Accuracy: ", accuracy(w, X_tst, Y_tst, 0.5))

	# with open(pkl_filename, 'wb') as file:
	#     pickle.dump(model, file)

exit()
poi = np.array([X_tst[0]])

start = timer()
exp = explain(model, poi).feasible_recourse_actions(X_trn, Y_trn, 10)
dt = timer() - start
print("Feasible Recourse Time:" + str(dt))

start = timer()
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
print(exp.Deegan_Packel_sample(1e-1, 1e-2))
dt = timer() - start
print("Explanation Time:" + str(dt))






