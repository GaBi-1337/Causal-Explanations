import pandas as pd
import numpy as np
from explain import Causal_Explanations, Mapper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    data = pd.read_csv("data/german.data", sep=" ", header=None) # read data
    data = data.rename(columns={key: value for value, key in enumerate(data.columns)}) # changed column names to numeric values
    data[20].replace(2, 0, inplace=True)

    #Get Transformed data from Mapper
    mapr = Mapper(data)
    data = mapr.get_data() 

    X = data[:, : -1]
    Y = data[:, -1]
    np.random.seed(0)
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=0.333, shuffle=True)

    # Train a model 
    model = RandomForestClassifier(n_estimators=50, max_features=None, n_jobs=1, random_state=0).fit(X_trn, Y_trn)
    print(model.score(X_trn, Y_trn), model.score(X_tst, Y_tst)) # model performance on Train and Test set respecitvely
    
    # Get the point of interest
    poi = mapr.point_inverse(X_tst[0])

    # Create baselines
    baselines = []
    for age in [20, 25, 30]:
        baseline = ['A13', np.max(data[1]), 'A30', 'A43', np.min(data[4]), 'A64', 'A75', np.min(data[7]), 'A94', 'A103', np.max(data[10]), 'A121', age, 'A143', 'A152', np.min(data[15]), 'A174', np.min(data[17]), 'A192', 'A202']
        baselines.append(baseline)
    baselines = np.array(baselines)

    exp = Causal_Explanations(model, poi, mapr, baselines)

    #Pick a Power index
    print(exp.Deegan_Packel_index()) 
    print(exp.Deegan_Packel_sample(1e-2, 1e-4, num_processes=4)) # Sampling version of index

if __name__ == "__main__":
    main()
