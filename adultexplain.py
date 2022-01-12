import pandas as pd
import numpy as np
from explain import Causal_Explanations, Mapper
import sklearn
import sklearn.tree
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
from itertools import combinations
# import shap

def computeNecSuf(exp, index_values):
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

# def computeNecSufSHAP(baselines, index_values, poi, model):
#     k2n, k2s, k3n, k3s, k5n, k5s = 0, 0, 0, 0, 0, 0
#     klist = [2, 3, 5]
#     n = len(index_values)
#     prediction = model.predict(reshape(1, -1))
#     for k in klist:
#         values = np.copy(index_values)
#         limit, limitdash = [0]*n, [0]*n
#         counter, counterdash = [0]*n, [0]*n
#         S = np.zeros(n)
#         Sdash = np.ones(n)
#         for kprime in range(k):
#             i = np.argmax(values)
#             S[i] = 1
#             Sdash[i] = 0
#             limit[i] = len(baselines[i])
#             values[i] = -10000

#         for i in range(n):
#             if(Sdash[i] == 1):
#                 limitdash[i] = len(baselines[i])

#         while(counter[0] <= limit[0]):
#             x = poi.copy()
#             for i in range(n):
#                 if(S[i] == 1):
#                     x[i] = baselines[i][counter[i]]
#             pred_new = model.predict(x.reshape(1, -1))
#             if(pred_new != prediction):




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
    print(len(X_trn))
    exit()

    # baselines_ohe_0 = []
    # # print(len(X[0, :]))
    # for i in range(len(X[0, :])):
    #     if(i == 0):
    #         baselines_ohe_0.append([20, 30, 85])
    #         continue
    #     if(i == 7):
    #         baselines_ohe_0.append([np.max(data[2])])
    #         continue
    #     if(i == 37):
    #         baselines_ohe_0.append([np.max(data[8])])
    #         continue
    #     if(i == 38):
    #         baselines_ohe_0.append([1000, 2500])
    #         continue
    #     if(i == 39):
    #         baselines_ohe_0.append([np.max(data[10])])
    #         continue
    #     baselines_ohe_0.append([0, 1])

    # baselines_ohe_1 = []
    # for i in range(len(X[0, :])):
    #     if(i == 0):
    #         baselines_ohe_1.append([20, 30, 85])
    #         continue
    #     if(i == 7):
    #         baselines_ohe_1.append([np.min(data[2])])
    #         continue
    #     if(i == 37):
    #         baselines_ohe_1.append([np.min(data[8])])
    #         continue
    #     if(i == 38):
    #         baselines_ohe_1.append([1000, np.max(data[9])])
    #         continue
    #     if(i == 39):
    #         baselines_ohe_1.append([np.min(data[10])])
    #         continue
    #     baselines_ohe_1.append([0, 1])

    index_values = {'JI': None, 'DPI': None, 'HPI': None, 'RI': None, 'BI': None, 'SI': None}
    key_pairs = list(combinations(index_values.keys(), 2))
    files = [open('rankings/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    ktfiles = [open('scores/' + pair[0] + '_' + pair[1] + '.txt', 'w') for pair in key_pairs]
    JI_file = open('nesuf_scores/JI.txt', 'w')
    DPI_file = open('nesuf_scores/DPI.txt', 'w')
    HPI_file = open('nesuf_scores/HPI.txt', 'w')
    RI_file = open('nesuf_scores/RI.txt', 'w')
    BI_file = open('nesuf_scores/BI.txt', 'w')
    SI_file = open('nesuf_scores/SI.txt', 'w')
    # SHAP_file = open('nesuf_scores/SHAP.txt', 'w')
    # LIME_file = open('nesuf_scores/LIME.txt', 'w')
    np.random.seed(0)
    np.random.shuffle(X_tst)
    
    count = 0
    for point in X_tst[:1000]:
        print(count, end = '\r')
        count += 1

        exp = Causal_Explanations(model, poi:=mapr.point_inverse(point), mapr, baselines_0 if (prediction:=model.predict(point.reshape(1, -1))) == 0 else baselines_1)

        index_values['JI'] = exp.Johnston_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['JI'])
        JI_file.write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['DPI'] = exp.Deegan_Packel_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['DPI'])
        DPI_file.write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['HPI'] = exp.Holler_Packel_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['HPI'])
        HPI_file.write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['RI'] = exp.Responsibility_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['RI'])
        RI_file.write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['BI'] = exp.Banzhaf_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['BI'])
        BI_file.write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

        index_values['SI'] = exp.Shapley_Shubik_index()
        k2n, k2s, k3n, k3s, k5n, k5s = computeNecSuf(exp, index_values['SI'])
        SI_file.write(str(k2n) + ' ' + str(k2s) + ' ' + str(k3n) + ' ' + str(k3s) + ' ' + str(k5n) + ' ' + str(k5s) + '\n')

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