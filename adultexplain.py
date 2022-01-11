import pandas as pd
import numpy as np
from explain import Causal_Explanations, Mapper
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
from itertools import combinations

def main():
    train = pd.read_csv("data/adult.data", header=None, na_values= ' ?')
    test = pd.read_csv("data/adult.test", header=None, na_values= ' ?') 
    train = train.dropna()
    test = test.dropna()
    train.drop([2, 3, 13], axis=1, inplace=True)
    test.drop([2, 3, 13], axis=1, inplace=True)
    data = pd.concat([train, test], ignore_index=True)
    data = data.rename(columns={key: value for value, key in enumerate(data.columns)})
    mapr = Mapper(data)

    baselines_0 = []
    for age in [20, 30, 85]:
        for ms in [' Never-married', ' Married-civ-spouse']:
            for caploss in [1000, 2500]:
                    baseline = [age, ' Private', np.max(data[2]), ms, ' Exec-managerial', ' Husband', ' White', ' Female', np.max(data[8]), caploss, np.max(data[10])]
                    baselines_0.append(baseline)
    baselines_0 = np.array(baselines_0)

    baselines_1 = []
    for age in [20, 30, 85]:
        for ms in [' Never-married', ' Married-civ-spouse']:
            for caploss in [1000, np.max(data[9])]:
                baseline = [age, ' Self-emp-not-inc', np.min(data[2]), ms, ' Other-service', ' Other-relative', ' White', ' Female', np.min(data[8]), caploss, np.min(data[10])]
                baselines_1.append(baseline)
    baselines_1 = np.array(baselines_1)

    data = mapr.get_data()
    X = data[:, : -1]
    Y = data[:, -1]
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.333, shuffle=False)
    model = sklearn.tree.DecisionTreeClassifier(max_depth = 5, random_state = 0).fit(X_trn, Y_trn)
    print(model.score(X_trn, Y_trn), model.score(X_tst, Y_tst))

    index_values = {'JI': None, 'DPI': None, 'HPI': None, 'RI': None, 'BI': None, 'SI': None}
    key_pairs = list(combinations(index_values.keys(), 2))
    files = [open('rankings/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    ktfiles = [open('scores/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    np.random.seed(0)
    np.random.shuffle(X_tst)
    
    for point in X_tst[:1000]:
        exp = Causal_Explanations(model, poi:=mapr.point_inverse(point), mapr, baselines_0 if (prediction:=model.predict(point.reshape(1, -1))) == 0 else baselines_1)
        index_values['JI'] = exp.Johnston_index()
        index_values['DPI'] = exp.Deegan_Packel_index()
        index_values['HPI'] = exp.Holler_Packel_index()
        index_values['RI'] = exp.Responsibility_index()
        index_values['BI'] = exp.Banzhaf_index()
        index_values['SI'] = exp.Shapley_Shubik_index()
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
    
if __name__ == "__main__":
    main()