import os
import pickle
import itertools

import numpy as np

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError

from sklearn.datasets import make_classification

"""
From DES matrix to prediction
"""
def predict_with_meta_dataset(ensemble, Y_test_meta, X_test, features=None):
    Y_pred = []
    ensemble = np.array(ensemble)
    for id_test, sub_ensemble_idx in enumerate(Y_test_meta):
        if sum(sub_ensemble_idx) == 0:
            sub_ensemble = np.copy(ensemble)
            if features is not None:
                sub_feature = np.copy(features)
        else:
            sub_ensemble = ensemble[sub_ensemble_idx.astype(bool)]
            if features is not None:
                sub_feature = [features[i] for i in range(len(features)) if sub_ensemble_idx.astype(bool)[i]]
        x_test = X_test[id_test, :]
        if features is None:
            y_preds = [clf.predict(x_test.reshape(1, -1))[0] for clf in sub_ensemble]
        else:
            y_preds = [clf.predict(x_test.reshape(1, -1)[:, f])[0] for clf, f in zip(sub_ensemble, sub_feature)]
        y_pred = np.argmax(np.bincount(y_preds))
        Y_pred.append(y_pred)
    return np.array(Y_pred)

"""
Check model is fitted
"""
def check_model_is_fitted(model, X_test):
    """
    Take X_test as small as possible
    """
    try:
        model.predict(X_test)
        return True
    except NotFittedError:
        return False
    
"""
Ensemble generation
"""
def first_generation(X, y, seed=None):
    mlp_parameters = list(itertools.product([1, 2, 4, 8, 16], [0, 0.2, 0.5, 0.9], [0.3, 0.6]))
    mlp_clf = [MLPClassifier(hidden_layer_sizes=(h,), momentum=m, learning_rate_init=a)
               for (h, m, a) in mlp_parameters]
    mlp_name = ['mlp_{0}_{1}_{2}'.format(*param) for param in mlp_parameters]

    neigbhors_number = [int(i) for i in np.linspace(1, X.shape[0], 20)]
    weighting_methods = ['uniform', 'distance', lambda x: abs(1-x)]
    knn_clf = [KNeighborsClassifier(n_neighbors=nn, weights=w)
               for (nn, w) in itertools.product(neigbhors_number, weighting_methods)]
    knn_name = ['knn_{0}_{1}'.format(*param)
                for param in itertools.product(neigbhors_number, ['uniform', 'distance', 'similarity'])]

    C = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
    degree = [2, 3]
    gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
    svm_clf_poly = [SVC(C=c, kernel='poly', degree=d) for (c, d) in itertools.product(C, degree)]
    svm_clf_poly_name = ['svm_poly_{0}_{1}'.format(*param) for param in itertools.product(C, degree)]
    svm_clf_rbf = [SVC(C=c, kernel='rbf', gamma=g) for (c, g) in itertools.product(C, gamma)]
    svm_clf_rbf_name = ['svm_rbf_{0}_{1}'.format(*param) for param in itertools.product(C, gamma)]

    dt_max_depth_params = list(itertools.product(['gini', 'entropy'], [1, 2, 3, 4, None]))
    dt_max_depth_clf = [DecisionTreeClassifier(criterion=c, max_depth=d) \
                        for (c, d) in dt_max_depth_params]
    dt_max_depth_name = ['dt_max_depth_{0}_{1}'.format(*param) for param in dt_max_depth_params]

    dt_max_features_params = list(itertools.product(['gini', 'entropy'], [None, 'sqrt', 'log2', 0.5]))
    dt_max_features_clf = [DecisionTreeClassifier(criterion=c, max_features=f) \
                           for (c, f) in dt_max_features_params]
    dt_max_features_name = ['dt_max_features_{0}_{1}'.format(*param) for param in dt_max_features_params]

    dt_min_leaf_params = [2, 3]
    dt_min_leaf_clf = [DecisionTreeClassifier(min_samples_leaf=l) for l in dt_min_leaf_params]
    dt_min_leaf_name = ['dt_min_leaf_{0}'.format(param) for param in dt_min_leaf_params]

    pool = mlp_clf + knn_clf + svm_clf_poly + svm_clf_rbf + dt_max_depth_clf + dt_max_features_clf + \
           dt_min_leaf_clf
    pool_name = mlp_name + knn_name + svm_clf_poly_name + svm_clf_rbf_name + dt_max_depth_name + \
                dt_max_features_name + dt_min_leaf_name

    ensemble = VotingClassifier(estimators=list(zip(pool_name, pool)))
    ensemble.fit(X, y)
    estimators = ensemble.estimators_

    return estimators, pool_name

def second_generation(X, y, seed=None):
    features = []
    ### 25 x 2 bagged trees
    bag_gini = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'),
                                 n_estimators=25,
                                 random_state=seed)
    bag_gini.fit(X, y)
    bag_gini_names = ['bag_gini_' + str(i) for i in range(25)]
    features.extend([np.arange(X.shape[1]) for _ in range(len(bag_gini_names))])

    bag_entropy = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),
                                 n_estimators=25,
                                 random_state=3*seed**2)
    bag_entropy.fit(X, y)
    bag_entropy_names = ['bag_entropy_' + str(i) for i in range(25)]
    features.extend([np.arange(X.shape[1]) for _ in range(len(bag_entropy_names))])

    ### 25 x 2 random subspaces
    rs_gini = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'),
                                n_estimators=25,
                                max_features=int(np.sqrt(X.shape[1])),
                                bootstrap=False,
                                random_state=seed)
    rs_gini.fit(X, y)
    rs_gini_names = ['rs_gini_' + str(i) for i in range(25)]
    features.extend(rs_gini.estimators_features_)

    rs_entropy = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),
                                   n_estimators=25,
                                   max_features=int(np.sqrt(X.shape[1])),
                                   bootstrap=False,
                                   random_state=3*seed**2)
    rs_entropy.fit(X, y)
    rs_entropy_names = ['rs_entropy_' + str(i) for i in range(25)]
    features.extend(rs_entropy.estimators_features_)

    ### 14 Ada
    nb_stumps = [2, 4, 8, 16, 32, 64, 128]
    ada_st_gini = [AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',
                                                                            max_depth=1),
                                      n_estimators=st,
                                      random_state=seed) for st in nb_stumps]
    ada_st_gini_names = ['ada_st_gini_' + str(i) for i in nb_stumps]
    features.extend([np.arange(X.shape[1]) for _ in range(len(ada_st_gini_names))])
    for clf in ada_st_gini:
        clf.fit(X, y)

    ada_st_entropy = [AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                               max_depth=1),
                                         n_estimators=st,
                                         random_state=3*seed**2) for st in nb_stumps]
    ada_st_entropy_names = ['ada_st_entropy_' + str(i) for i in nb_stumps]
    features.extend([np.arange(X.shape[1]) for _ in range(len(ada_st_entropy_names))])
    for clf in ada_st_entropy:
        clf.fit(X, y)

    ### 8 Ada DT
    nb_dt = [2, 4, 8, 16]
    ada_dt_gini = [AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',
                                                                            max_depth=3),
                                      n_estimators=dt,
                                      random_state=seed) for dt in nb_dt]
    ada_dt_gini_names = ['ada_dt_gini_' + str(i) for i in nb_dt]
    features.extend([np.arange(X.shape[1]) for _ in range(len(ada_dt_gini_names))])
    for clf in ada_dt_gini:
        clf.fit(X, y)

    ada_dt_entropy = [AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                               max_depth=3),
                                         n_estimators=st,
                                         random_state=3*seed**2) for dt in nb_dt]
    ada_dt_entropy_names = ['ada_dt_entropy_' + str(i) for i in nb_dt]
    features.extend([np.arange(X.shape[1]) for _ in range(len(ada_dt_entropy_names))])
    for clf in ada_dt_entropy:
        clf.fit(X, y)

    ### 24 ANN
    mlp_parameters = list(itertools.product([1, 2, 4, 8, 32, 128],\
                                            [0, 0.2, 0.5, 0.9]))
    mlp_clf = [MLPClassifier(hidden_layer_sizes=(h,), momentum=m)
               for (h, m) in mlp_parameters]
    for clf in mlp_clf:
        clf.fit(X, y)
    mlp_name = ['mlp_{0}_{1}'.format(*param) for param in mlp_parameters]
    features.extend([np.arange(X.shape[1]) for _ in range(len(mlp_name))])

    ### 54 SVM
    C = np.logspace(-3, 2, num=6)
    gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]

    svm_linear = [SVC(C=c, kernel='poly', degree=1) for c in C]
    for clf in svm_linear:
        clf.fit(X, y)
    svm_linear_names = ['svm_linear_'+str(c) for c in C]
    features.extend([np.arange(X.shape[1]) for _ in range(len(svm_linear_names))])

    svm_rbf = [SVC(C=c, gamma=g) for c, g in itertools.product(C, gamma)]
    for clf in svm_rbf:
        clf.fit(X, y)
    svm_rbf_names = ['svm_rbf_{0}_{1}'.format(*param) for param in itertools.product(C, gamma)]
    features.extend([np.arange(X.shape[1]) for _ in range(len(svm_rbf_names))])

    pool = bag_gini.estimators_ + bag_entropy.estimators_ + rs_gini.estimators_ + rs_entropy.estimators_ + \
           ada_st_gini + ada_st_entropy + ada_dt_gini + ada_dt_entropy + mlp_clf + svm_linear + svm_rbf

    pool_name = bag_gini_names + bag_entropy_names + rs_gini_names + rs_entropy_names + ada_st_gini_names + \
                ada_st_entropy_names + ada_dt_gini_names + ada_dt_entropy_names + mlp_name + svm_linear_names + \
                svm_rbf_names

    return pool, pool_name, features

def third_generation(X, y, size=200, seed=None):
    mlp_parameters = list(itertools.product([1, 2, 4, 8, 32, 128],\
                                            [0, 0.2, 0.5, 0.9],
                                            [0.1, 0.3, 0.6]))
    mlp_clf = [MLPClassifier(hidden_layer_sizes=(h,), momentum=m, learning_rate_init=a)
               for (h, m, a) in mlp_parameters]
    mlp_name = ['mlp_{0}_{1}_{2}'.format(*param) for param in mlp_parameters]

    neigbhors_number = [int(i) for i in np.linspace(1, X.shape[0], 40)]
    weighting_methods = ['uniform', 'distance']
    knn_clf = [KNeighborsClassifier(n_neighbors=nn, weights=w)
               for (nn, w) in itertools.product(neigbhors_number, weighting_methods)]
    knn_name = ['knn_{0}_{1}'.format(*param)
                for param in itertools.product(neigbhors_number, ['uniform', 'distance'])]
    C = np.logspace(-3, 7, num=11)
    degree = [2, 3, 4]
    gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
    svm_clf_poly = [SVC(C=c, kernel='poly', degree=d) for (c, d) in itertools.product(C, degree)]
    svm_clf_poly_name = ['svm_poly_{0}_{1}'.format(*param) for param in itertools.product(C, degree)]
    svm_clf_rbf = [SVC(C=c, kernel='rbf', gamma=g) for (c, g) in itertools.product(C, gamma)]
    svm_clf_rbf_name = ['svm_rbf_{0}_{1}'.format(*param) for param in itertools.product(C, gamma)]

    dt_params = list(itertools.product(['gini', 'entropy'], \
                                       [1, 2, 3, 4, 5, None], \
                                       [None, 'sqrt', 'log2'], \
                                       ['best', 'random']))
    dt_clf = [DecisionTreeClassifier(criterion=c, max_depth=d, max_features=f, splitter=s)
              for (c, d, f, s) in dt_params]
    dt_name = ['dt_{0}_{1}_{2}_{3}'.format(*param) for param in dt_params]

    et_clf = [ExtraTreeClassifier(criterion=c, max_depth=d, max_features=f, splitter=s)
              for (c, d, f, s) in dt_params]
    et_name = ['et_{0}_{1}_{2}_{3}'.format(*param) for param in dt_params]

    ada_params = list(itertools.product([2**i for i in range(1, 14)], \
                                        [1, 2, 3]))
    ada_dt_clf = [AdaBoostClassifier(n_estimators=n, base_estimator=DecisionTreeClassifier(max_depth=m))
                  for (n, m) in ada_params]
    ada_et_clf = [AdaBoostClassifier(n_estimators=n, base_estimator=ExtraTreeClassifier(max_depth=m))
                  for (n, m) in ada_params]
    ada_dt_name = ['ada_dt_{0}_{1}'.format(*param) for param in ada_params]
    ada_et_name = ['ada_et_{0}_{1}'.format(*param) for param in ada_params]

    nb_bag_est = 50
    nb_bag_stumps = 200
    bag_dt = BaggingClassifier(n_estimators=nb_bag_est, base_estimator=DecisionTreeClassifier())
    bag_et = BaggingClassifier(n_estimators=nb_bag_est, base_estimator=ExtraTreeClassifier())
    bag_stumps = BaggingClassifier(n_estimators=nb_bag_stumps, base_estimator=DecisionTreeClassifier(max_depth=1))
    bag_dt.fit(X, y)
    bag_et.fit(X, y)
    bag_stumps.fit(X, y)
    dt_bag_clf = bag_dt.estimators_
    et_bag_clf = bag_et.estimators_
    stump_bag_clf = bag_stumps.estimators_
    dt_bag_name = ['dt_bag_{0}'.format(nb_est) for nb_est in range(nb_bag_est)]
    et_bag_name = ['et_bag_{0}'.format(nb_est) for nb_est in range(nb_bag_est)]
    stump_bag_name = ['stump_bag_{0}'.format(nb_est) for nb_est in range(nb_bag_stumps)]

    bag_dt_clf = [bag_dt]
    bag_et_clf = [bag_dt]
    bag_stump_clf = [bag_stumps]
    bag_dt_name = ['bag_dt_{0}'.format(str(nb_bag_est))]
    bag_et_name = ['bag_et_{0}'.format(str(nb_bag_est))]
    bag_stump_name = ['bag_stump_{0}'.format(str(200))]

    nb_rf = 15
    rf = RandomForestClassifier(n_estimators=nb_rf)
    rf.fit(X, y)
    dt_rf_clf = rf.estimators_
    dt_rf_name = ['dt_rf_{0}'.format(nb_est) for nb_est in range(nb_rf)]

    log_parameters = list(itertools.product(['l1', 'l2'],\
                                            np.logspace(-5, 9, num=15),
                                            [True, False]))
    log_clf = [LogisticRegression(penalty=l, C=c, fit_intercept=f) for (l, c, f) in log_parameters]
    log_name = ['log_{0}_{1}_{2}'.format(*param) for param in log_parameters]


    sgd_parameters = list(itertools.product(['hinge',
                                             'log',
                                             'modified_huber',
                                             'squared_hinge',
                                             'perceptron',
                                             'squared_loss',
                                             'huber',
                                             'epsilon_insensitive',
                                             'squared_epsilon_insensitive'],
                                            ['elasticnet'],
                                            [True, False],
                                            np.arange(0, 1.1, 0.1)))
    sgd_clf = [SGDClassifier(loss=l, penalty=p, fit_intercept=f, l1_ratio=l1) for (l, p, f, l1) in sgd_parameters]
    sgd_name = ['sgd_{0}_{1}_{2}_{3}'.format(*param) for param in sgd_parameters]

    pool = mlp_clf + knn_clf + svm_clf_poly + svm_clf_rbf + dt_clf + et_clf + ada_dt_clf + ada_et_clf + \
                dt_bag_clf + et_bag_clf + stump_bag_clf + bag_dt_clf + bag_et_clf + bag_stump_clf + dt_rf_clf + \
                log_clf + sgd_clf
    pool_name = mlp_name + knn_name + svm_clf_poly_name + svm_clf_rbf_name + dt_name + et_name + ada_dt_name + \
                ada_et_name + dt_bag_name + et_bag_name + stump_bag_name + bag_dt_name + bag_et_name + \
                bag_stump_name + dt_rf_name + log_name + sgd_name

    for model in pool:
        if not check_model_is_fitted(model, X[0, :].reshape((1, -1))):
            model.fit(X, y)

    np.random.seed(seed)
    order = np.random.permutation(range(len(pool)))
    estimators = [pool[i] for i in order[:size]]

    return estimators, pool_name	






