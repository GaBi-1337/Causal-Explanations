import pandas as pd
import numpy as np
from explain import Causal_Explanations, Mapper
import sklearn
import sklearn.tree
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
from itertools import combinations

def computeNecSuf(exp, index_values):
    '''
    Computes the Necessity and Sufficiency scores
    '''
    n = len(index_values)
    values = np.copy(index_values)
    S2, S3, S5 = np.zeros(n), np.zeros(n), np.zeros(n)

    i = np.argmax(values)
    S2[i] = 1
    S3[i] = 1
    S5[i] = 1
    values[i] = -10000

    i = np.argmax(values)
    S2[i] = 1
    S3[i] = 1
    S5[i] = 1
    values[i] = -10000

    i = np.argmax(values)
    S3[i] = 1
    S5[i] = 1
    values[i] = -10000

    i = np.argmax(values)
    S5[i] = 1
    values[i] = -10000

    i = np.argmax(values)
    S5[i] = 1
    values[i] = -10000

    return exp.value(S2), 1 - exp.value(1-S2), exp.value(S3), 1 - exp.value(1-S3), exp.value(S5), 1 - exp.value(1-S5)

def main():
    # Get data
    train = pd.read_csv("data/adult.data", header=None, na_values= ' ?')
    test = pd.read_csv("data/adult.test", header=None, na_values= ' ?') 
    train = train.dropna()
    test = test.dropna()
    train.drop([2, 3, 13], axis=1, inplace=True)
    test.drop([2, 3, 13], axis=1, inplace=True)
    data = pd.concat([train, test], ignore_index=True)
    data = data.rename(columns={key: value for value, key in enumerate(data.columns)})
    mapr = Mapper(data)
    data = mapr.get_data()
    X = data[:, : -1]
    Y = data[:, -1]
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.333, shuffle=False)
    
    # Create model
    model = sklearn.tree.DecisionTreeClassifier(max_depth = 5, random_state = 0).fit(X_trn, Y_trn)
    print(model.score(X_trn, Y_trn), model.score(X_tst, Y_tst))

    # Create baselines for negative outcomes
    baselines_0 = []
    for age in [20, 30, 85]:
        for ms in [' Never-married', ' Married-civ-spouse']:
            for caploss in [1000, 2500]:
                    baseline = [age, ' Private', np.max(data[2]), ms, ' Exec-managerial', ' Husband', ' White', ' Female', np.max(data[8]), caploss, np.max(data[10])]
                    baselines_0.append(baseline)
    baselines_0 = np.array(baselines_0)

    # Create baseline for positive outcomes
    baselines_1 = []
    for age in [20, 30, 85]:
        for ms in [' Never-married', ' Married-civ-spouse']:
            for caploss in [1000, np.max(data[9])]:
                baseline = [age, ' Self-emp-not-inc', np.min(data[2]), ms, ' Other-service', ' Other-relative', ' White', ' Female', np.min(data[8]), caploss, np.min(data[10])]
                baselines_1.append(baseline)
    baselines_1 = np.array(baselines_1)

    # Dictionary for index computations
    index_values = {'JI': None, 'DPI': None, 'HPI': None, 'RI': None, 'BI': None, 'SI': None}

    #open files to write computations
    key_pairs = list(combinations(index_values.keys(), 2))
    files = [open('rankings/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    ktfiles = [open('scores/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    nsfiles = {'JI': open('nesuf_scores/JI.txt', 'w'), 'DPI': open('nesuf_scores/DPI.txt', 'w'), 'HPI': open('nesuf_scores/HPI.txt', 'w'), 'RI': open('nesuf_scores/RI.txt', 'w'), 'BI': open('nesuf_scores/BI.txt', 'w'), 'SI': open('nesuf_scores/SI.txt', 'w')}

    np.random.seed(0)
    np.random.shuffle(X_tst)
    
    for point in X_tst[:1000]:

        exp = Causal_Explanations(model, poi:=mapr.point_inverse(point), mapr, baselines_0 if (prediction:=model.predict(point.reshape(1, -1))) == 0 else baselines_1)

        index_values['JI'] = exp.Johnston_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['JI'])
        nsfiles['JI'].write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['DPI'] = exp.Deegan_Packel_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['DPI'])
        nsfiles['DPI'].write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['HPI'] = exp.Holler_Packel_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['HPI'])
        nsfiles['HPI'].write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['RI'] = exp.Responsibility_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['RI'])
        nsfiles['RI'].write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['BI'] = exp.Banzhaf_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['BI'])
        nsfiles['BI'].write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['SI'] = exp.Shapley_Shubik_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['SI'])
        nsfiles['SI'].write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        for key_pair, file, kt in zip(key_pairs, files, ktfiles):
            score = kendalltau(index_values[key_pair[0]], index_values[key_pair[1]], variant='b')[0]
            kt.write(str(score) + '\n')
            if score != 1.0:
                file.write("Point: " + str(poi) + '\n')
                file.write("Prediction: " + str(prediction) + '\n')
                file.write(key_pair[0] + ": " + str(index_values[key_pair[0]]) + '\n')
                file.write(key_pair[1] + ": " + str(index_values[key_pair[1]]) + '\n')
                file.write("Tau: " + str(score) + '\n')
                file.write('\n')

    for file in files: file.close()
    for file in ktfiles: file.close()
    for file in nsfiles: nsfiles[file].close()
    
if __name__ == "__main__":
    main()