from math import remainder
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def Kfold_cv_accuracy(X_trn, Y_trn, model):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    accurates = list()
    for train, validate in kf.split(X_trn):
        model.fit(X_trn[train], Y_trn[train])
        predicted = model.predict(X_trn[validate])
        Y_validate = Y_trn[validate]
        accurates.append(np.mean([1 if predicted[y] == Y_validate[y] else 0 for y in range(len(predicted))])) 
    return np.mean(accurates)

def best_trained_DT(X_trn, Y_trn):
    accuracy = dict()
    for depth in [1, 3, 6, 9, 12, 15, 18, 21]:
        accuracy[depth] = Kfold_cv_accuracy(X_trn, Y_trn, DecisionTreeClassifier(max_depth=depth))
    return DecisionTreeClassifier(max_depth = max(accuracy, key=accuracy.get)).fit(X_trn, Y_trn)

def best_trained_KNN(X_trn, Y_trn):
    accuracy = dict()
    for K in [1, 4, 8, 12, 16, 20, 24, 28, 32]:
        accuracy[K] = Kfold_cv_accuracy(X_trn, Y_trn, KNeighborsClassifier(n_neighbors=K, n_jobs=-1))
    return KNeighborsClassifier(n_neighbors=max(accuracy, key=accuracy.get), n_jobs=-1).fit(X_trn, Y_trn)

def best_trained_SVM(X_trn, Y_trn):
    accuracy = dict()
    for λ in [0.0001, 0.01, 0.1, 1, 10, 100]:
        accuracy[λ] = Kfold_cv_accuracy(X_trn, Y_trn, LinearSVC(C=λ, random_state=0))
    return LinearSVC(C=max(accuracy, key=accuracy.get), random_state=0).fit(X_trn, Y_trn)

def best_trained_LM(X_trn, Y_trn):
    accuracy = dict()
    for λ in [0.0001, 0.01, 0.1, 1, 10, 100]:
        accuracy[λ] = Kfold_cv_accuracy(X_trn, Y_trn, LogisticRegression(C=λ, random_state=0, max_iter=1000, multi_class='ovr', n_jobs=-1))
    return LogisticRegression(C=max(accuracy, key=accuracy.get), random_state=0, max_iter=1000, multi_class='ovr', n_jobs=-1).fit(X_trn, Y_trn)

def best_trained_NB(X_trn, Y_trn):
    return GaussianNB().fit(X_trn, Y_trn)

def best_trained_RF(X_trn, Y_trn):
    accuracy = dict()
    for trees in [1, 12, 25, 50, 100, 200]:
        accuracy[trees] = Kfold_cv_accuracy(X_trn, Y_trn, RandomForestClassifier(n_estimators=trees, max_features=None, n_jobs=-1, random_state=0))
        print(accuracy[trees])
    return RandomForestClassifier(n_estimators=max(accuracy, key=accuracy.get), max_features=None, n_jobs=-1, random_state=0).fit(X_trn, Y_trn)

def main():
    
    """for model in models:
        predicted = model(X_trn, Y_trn).predict(X_tst)
        models[model] = np.mean([1 if predicted[y] == Y_tst[y] else 0 for y in range(len(predicted))])
    
    print(models.values())"""
    

if __name__ == "__main__":
    main()