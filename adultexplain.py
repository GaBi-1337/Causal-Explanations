import pandas as pd
import numpy as np
from data import Representer
from explain import Causal_Explanations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from scipy.stats import kendalltau
from itertools import combinations
def main():
    mp.set_start_method('loky')

    train = pd.read_csv("data/adult.data", header=None, na_values= ' ?')
    test = pd.read_csv("data/adult.test", header=None, na_values= ' ?') 
    train = train.dropna()
    test = test.dropna()
    train.drop([2, 3, 13], axis=1, inplace=True)
    test.drop([2, 3, 13], axis=1, inplace=True)
    data = pd.concat([train, test], ignore_index=True)
    data = data.rename(columns={key: value for value, key in enumerate(data.columns)})
    rep = Representer(data)

    baselines_0 = []
    for age in [20, 30, 40, 50]:
        for wc in [' Private', ' Federal-gov']:
            for ocup in [' Exec-managerial', ' Prof-specialty', ' Adm-clerical']:
                for sex in [' Male', ' Female']:
                    baseline = [age, wc, np.max(data[2]), ' Married-civ-spouse', ocup, ' Husband', ' White', sex, np.max(data[8]), np.min(data[9]), np.max(data[10])]
                    baselines_0.append(baseline)
    baselines_0 = np.array(baselines_0)

    baselines_1 = []
    for age in [20, 30, 40, 50]:
        for ms in [' Never-married', ' Widowed']:
            for ocup in [' Tech-support', ' Craft-repair', ' Handlers-cleaners']:
                for race in [' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Black']:
                    for sex in [' Male', ' Female']:
                        baseline = [age, ' Without-pay', np.min(data[2]), ms, ocup, ' Unmarried', race, sex, np.min(data[8]), np.max(data[9]), np.min(data[10])]
                        baselines_1.append(baseline)
    baselines_1 = np.array(baselines_1)

    data = rep.get_data()
    X = data[:, : -1]
    Y = data[:, -1]
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.333, shuffle=False)
    model = RandomForestClassifier(n_estimators=50, max_features=None, n_jobs=1, random_state=0).fit(X_trn, Y_trn)
    print(model.score(X_trn, Y_trn), model.score(X_tst, Y_tst))

    index_values = {'JI': None, 'DPI': None, 'HPI': None, 'RI': None, 'BI': None, 'SI': None}
    key_pairs = list(combinations(index_values.keys(), 2))
    files = [open('rankings/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    ktfiles = [open('scores/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    np.random.seed(0)
    np.random.shuffle(X_tst)    
    for point in X_tst[:2]:
        exp = Causal_Explanations(model, poi:=rep.point_inverse(point), rep, baselines_0 if (prediction:=model.predict(point.reshape(1, -1))) == 0 else baselines_1)
        index_values['JI'] = exp.Johnston_sample(1e-1, 1e-2, num_processes = 5)
        index_values['DPI'] = exp.Deegan_Packel_sample(1e-1, 1e-2, num_processes = 5)
        index_values['HPI'] = exp.Holler_Packel_sample(1e-1, 1e-2, num_processes = 5)
        index_values['RI'] = exp.Responsibility_sample(1e-1, 1e-2, num_processes = 5)
        index_values['BI'] = exp.Banzhaf_sample(1e-1, 1e-2, num_processes = 5)
        index_values['SI'] = exp.Shapley_Shubik_sample(1e-1, 1e-2, num_processes = 5)
        for key_pair, file, kt in zip(key_pairs, files, ktfiles):
            score = kendalltau(index_values[key_pair[0]], index_values[key_pair[1]], variant='c')[0]
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
