import os
import xgboost
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import shap
from flow import GraphExplainer, CausalLinks, translator, build_feature_graph, create_xgboost_f, node_dict2str_dict, edge_credits2edge_credit, group_nodes

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

if __name__ == "__main__":
    np.random.seed(42)
    #Nutrition flow
    X,y = shap.datasets.nhanesi()
    X_display,y_display = shap.datasets.nhanesi(display=True)
    X = X.iloc[:, 1:]
    X_display = X_display.iloc[:, 1:]
    xgb_full = xgboost.DMatrix(X, label=y)
    n_bg = 1
    nsamples = 100
    nruns = 100
    bg = X.fillna(X.mean()).sample(n_bg)
    fg = X[:nsamples]
    sample_ind = 55
    model = xgboost.train({"eta": 0.002,"max_depth": 3, "objective": "survival:cox","subsample": 0.5}, xgb_full, 1000, evals = [(xgb_full, "test")], verbose_eval=1000)
    causal_links = CausalLinks()
    categorical_feature_names = ['Race', 'Sex']
    display_translator = translator(X.columns, X, X_display)
    target_name = 'predicted hazard'
    feature_names = list(X.columns)
    causal_links.add_causes_effects(feature_names, target_name, create_xgboost_f(feature_names, model, output_margin=True))
    A = ['Diastolic BP', 'Systolic BP']
    D = ['Pulse pressure']
    causal_links.add_causes_effects(A, D, lambda dbp, sdp: sdp-dbp)
    A = ['Age', 'Sex', 'Poverty index', 'Race']
    D = list(set(feature_names) - set(A) - set(['Pulse pressure']))
    causal_links.add_causes_effects(A, D)
    A = ['Age', 'Sex', 'Race']
    D = ['Poverty index']
    causal_links.add_causes_effects(A, D)
    causal_graph = build_feature_graph(X.fillna(X.mean()), causal_links, categorical_feature_names, display_translator, target_name)
    E = GraphExplainer(causal_graph, bg[:1])
    E.prepare_graph(fg)
    G = copy.deepcopy(E.graph)
    G = group_nodes(G, [n for n in G if n.name in ['White blood cells', 'Sedimentation rate']], 'Inflamation')
    G = group_nodes(G, [n for n in G if n.name in ['Systolic BP', 'Diastolic BP']], 'Blood pressure')
    G = group_nodes(G, [n for n in G if n.name in ['TS', 'TIBC', 'Serum Iron']], 'Iron')
    G = group_nodes(G, [n for n in G if n.name in ['Serum Protein', 'Serum Albumin']], 'Blood protein')
    causal_edge_credits = []
    for i in range(len(bg)):
        E = GraphExplainer(causal_graph, bg[i:i+1])
        E.prepare_graph(fg)
        G = copy.deepcopy(E.graph)
        G = group_nodes(G, [n for n in G if n.name in ['White blood cells', 'Sedimentation rate']], 'Inflamation')
        G = group_nodes(G, [n for n in G if n.name in ['Systolic BP', 'Diastolic BP']], 'Blood pressure')
        G = group_nodes(G, [n for n in G if n.name in ['TS', 'TIBC', 'Serum Iron']], 'Iron')
        G = group_nodes(G, [n for n in G if n.name in ['Serum Protein', 'Serum Albumin']], 'Blood protein')
        explainer = GraphExplainer(G, bg[i:i+1], nruns=nruns) 
        cf_c = explainer.shap_values(fg, skip_prepare=True)        
        causal_edge_credits.append(node_dict2str_dict(cf_c.edge_credit))
    cf_c.draw(sample_ind, max_display=10, show_fg_val=False, edge_credit=edge_credits2edge_credit(causal_edge_credits, cf_c.graph))

    #Adult flow
    X,y = shap.datasets.adult()
    X_display,y_display = shap.datasets.adult(display=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    xgb_train = xgboost.DMatrix(X_train, label=y_train)
    xgb_test = xgboost.DMatrix(X_test, label=y_test)
    n_bg = 1
    nsamples = 100
    nruns = 100
    bg = X.fillna(X.mean()).sample(n_bg)
    fg = X[:nsamples]
    sample_ind = 3
    model = xgboost.train({"eta": 0.01, "max_depth": 4, 'objective':'binary:logistic',"subsample": 0.9}, xgb_train, 3000, evals = [(xgb_test, "test")], verbose_eval=1000)
    causal_links = CausalLinks()
    categorical_feature_names = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    display_translator = translator(X.columns, X, X_display)
    target_name = 'predicted income >50k'
    feature_names = list(X.columns)
    causal_links.add_causes_effects(feature_names, target_name, create_xgboost_f(feature_names, model))
    A = ['Age', 'Sex', 'Country', 'Race']
    D = list(set(feature_names) - set(A))
    causal_links.add_causes_effects(A, D)
    causal_graph = build_feature_graph(X.fillna(X.mean()), causal_links, categorical_feature_names, display_translator, target_name, method='linear')
    causal_edge_credits = []
    for i in range(len(bg)):
        cf_c = GraphExplainer(causal_graph, bg[i:i+1], nruns=nruns).shap_values(fg)
        causal_edge_credits.append(node_dict2str_dict(cf_c.edge_credit))
    cf_c.draw(sample_ind, max_display=10, show_fg_val=False, edge_credit=edge_credits2edge_credit(causal_edge_credits, cf_c.graph))