import numpy as np
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

"""
Classifier Chain
"""
def train_chain(clf, X_train, Y_train):
    n_samples, n_classes = Y_train.shape
    X_temp = np.copy(X_train)
    chain = []
    for class_id in range(n_classes):
        if len(np.unique(Y_train[:, class_id])) == 1:
            classifier = DecisionTreeClassifier()
        else:
            classifier = clone(clf)
        classifier.fit(X_temp, Y_train[:, class_id])
        X_temp = np.hstack((X_temp, Y_train[:, class_id].reshape((-1, 1))))
        chain.append(classifier)
    return chain

def cc_predict(chain, X_test):
    n_classes = len(chain)
    n_samples = X_test.shape[0]
    X_temp = np.copy(X_test)
    Y_pred = np.zeros((n_samples, n_classes))
    for class_id in range(n_classes):
        classifier = chain[class_id]
        pred = classifier.predict(X_temp)
        Y_pred[:, class_id] = pred
        X_temp = np.hstack((X_temp, pred.reshape((-1, 1))))
    return Y_pred

"""
Probabilistic Classifier Chain
"""
def pcc_probability(chain, x_sample, pattern):
    n_classes = len(pattern)
    probability = 1
    x_temp = np.copy(x_sample)
    for class_id in range(n_classes):
        classifier = chain[class_id]
        proba = classifier.predict_proba(x_temp.reshape(1, -1))
        if len(proba) == 1:
            probability *= proba[0][0] # à changer qq part
        else:
            probability *= proba[0, pattern[class_id]]
        x_temp = np.hstack((x_temp, pattern[class_id]))
    return probability

def pcc_monte_carlo(chain, x_sample, seed=None):
    n_classes = len(chain)
    x_temp = np.copy(x_sample)
    pattern = []
    for class_id in range(n_classes):
        classifier = chain[class_id]
        proba_zero = classifier.predict_proba(x_temp.reshape(1, -1))[0][0]
        if seed is not None:
            random.seed(seed+class_id)
        if random.random() < proba_zero:
            pattern.append(0)
            x_temp = np.hstack((x_temp, 0))
        else:
            pattern.append(1)
            x_temp = np.hstack((x_temp, 1))
    return np.array(pattern)