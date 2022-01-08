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
    data = pd.read_csv("data/german.data", sep=" ", header=None)
    data = data.rename(columns={key: value for value, key in enumerate(data.columns)})
    data[20].replace(2, 0, inplace=True)
    rep = Representer(data)

    baselines_0 = []
    for age in [20, 25, 30]:
        baseline = ['A13', np.max(data[1]), 'A30', 'A43', np.min(data[4]), 'A64', 'A75', np.min(data[7]), 'A94', 'A103', np.max(data[10]), 'A121', age, 'A143', 'A152', np.min(data[15]), 'A174', np.min(data[17]), 'A192', 'A202']
        baselines_0.append(baseline)
    baselines_0 = np.array(baselines_0)

    baselines_1 = []
    for age in [20, 25, 30]:
        baseline = ['A11', np.min(data[1]), 'A34', 'A40', np.max(data[4]), 'A61', 'A71', np.max(data[7]), 'A93', 'A101', np.min(data[10]), 'A124', age, 'A141', 'A151', np.max(data[15]), 'A171', np.max(data[17]), 'A191', 'A201']
        baselines_1.append(baseline)
    baselines_1 = np.array(baselines_1)
    
    data = rep.get_data()
    X = data[:, : -1]
    Y = data[:, -1]
    np.random.seed(0)
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.333, shuffle=True)
    model = RandomForestClassifier(n_estimators=50, max_features=None, n_jobs=1, random_state=0).fit(X_trn, Y_trn)
    print(model.score(X_trn, Y_trn), model.score(X_tst, Y_tst))
    
    index_values = {'JI': None, 'DPI': None, 'HPI': None, 'RI': None, 'BI': None, 'SI': None}
    key_pairs = list(combinations(index_values.keys(), 2))
    files = [open('rankings/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    ktfiles = [open('scores/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    np.random.shuffle(X_tst)    
    for point in X_tst[:200]:
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
