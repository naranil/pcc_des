import os
import subprocess
import pickle
import platform
import time
import itertools
import argparse
import sys
import logging
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from des_tools import first_generation, second_generation, third_generation
from des_tools import predict_with_meta_dataset
from classifier_chain import train_chain, cc_predict, pcc_probability, pcc_monte_carlo
from utils import save_method_results
from datasets import load_datasets

from sklearn.metrics import accuracy_score
from mlc_tools import true_loss, precision_loss

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='ionosphere', type=str, help="Training dataset")
parser.add_argument("--iteration", default='0 1 2 3 4', type=str, help="Iterations")
parser.add_argument("--generation", default='first', type=str, help="Ensemble generation procedure")
parser.add_argument("--n_estimators", default=200, type=int, help="If third generation, number of models")
parser.add_argument("--mc", default=1000, type=int, help="Monte Carlo sampling")
parser.add_argument("--datasize", default=None, type=float, help="Take a subset of the dataset")
parser.add_argument("--verbose", default=True, type=bool, help="Display intermediate results")
parser.add_argument("--save", default=True, type=bool, help="Save important variables")
parser.add_argument("--sample", default=None, type=float, help="Reduce the size of the dataset")

args = parser.parse_args()

# Random seed
seed = 123

# Result path
result_path = './output/'
# Dataset
dataset = args.dataset
# Iterations
iterations = [int(iter_id) for iter_id in args.iteration.split(' ')]
# Subset of the dataset
datasize = args.datasize
# Verbose
verbose = args.verbose
save = args.save
# Generation type
generation = args.generation
# Number of models if third generation
n_estimators = args.n_estimators
# Monte Carlo sampling
mc = args.mc

#dataset = 'pcmac'
#iteration = '0 1 2 3 4'
#generation = 'first'
#mc = 1000
#datasize = 0.3
#save = True
#verbose = True

if __name__ == '__main__':
	data_path = os.path.join(result_path, dataset)
	if not os.path.isdir(data_path):
		os.makedirs(data_path)

	X, Y = load_datasets(dataset)
	X, Y = X.values, Y.values
	if datasize is not None:
		X, _, Y, _ = train_test_split(X, Y, train_size=datasize, stratify=Y, random_state=seed)

	scaler = MinMaxScaler()	
	X = scaler.fit_transform(X)

	m, n = X.shape

	for id_iter in iterations:
		print("DATASET: {0}, ITER: {1}".format(dataset, str(id_iter)))
		kf = KFold(n_splits=5, random_state=seed + id_iter)
		iter_path = os.path.join(data_path, 'iter_'+str(id_iter))
		if not os.path.isdir(iter_path):
			os.makedirs(iter_path)
		for id_cv, (train, test) in enumerate(kf.split(X)):
			print("DATASET: {0}, ITER: {1}, CV: {2}".format(dataset, 
															str(id_iter), 
															str(id_cv)))
			cv_path = os.path.join(iter_path, 'cv_'+str(id_cv))
			if not os.path.isdir(cv_path):
				os.makedirs(cv_path)
			X_train, X_test = X[train, :], X[test, :]
			Y_train, Y_test = Y[train], Y[test]
			X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25)

			# POOL GENERATION
			if generation == 'first':
				ensemble, _ = first_generation(X_train, Y_train, seed=seed)
				Y_val_meta = np.array([(model.predict(X_val) == Y_val).astype(int) for model in ensemble]).T
				Y_test_meta = np.array([(model.predict(X_test) == Y_test).astype(int) for model in ensemble]).T
			elif generation == 'second':
				ensemble, _ , features= second_generation(X_train, Y_train, seed=seed)
				Y_val_meta = np.array([(model.predict(X_val[:, f]) == Y_val).astype(int) \
									   for model, f in zip(ensemble, features)]).T
				Y_test_meta = np.array([(model.predict(X_test[:, f]) == Y_test).astype(int) \
										for model, f in zip(ensemble, features)]).T
			else:
				ensemble = third_generation(X_train, Y_train, size=n_estimators, seed=seed)
				Y_val_meta = np.array([(model.predict(X_val) == Y_val).astype(int) for model in ensemble]).T
				Y_test_meta = np.array([(model.predict(X_test) == Y_test).astype(int) for model in ensemble]).T
			
			p = len(ensemble)

			if save:
				save_method_results(cv_path, 
									X_train=X_train,
									Y_train=Y_train,
									X_test=X_test,
									Y_test=Y_test,
									X_val=X_val,
									Y_val=Y_val,
									Y_val_meta=Y_val_meta,
									Y_test_meta=Y_test_meta)			

			# META-LEARNING STEP
			ml_clf = OneVsRestClassifier(estimator=LogisticRegression())  
			ml_clf.fit(X_val, Y_val_meta)  
			proba = ml_clf.predict_proba(X_test)

			# ENSEMBLE
			Y_pred_meta = np.ones((X_test.shape[0], p))
			Y_pred = predict_with_meta_dataset(ensemble, Y_pred_meta, X_test)
			if save:
				save_method_results(cv_path, method_name='ensemble', Y_pred=Y_pred)
			if verbose:
				print("For ENSEMBLE: {0}".format(accuracy_score(Y_test, Y_pred)))

			# PRECISION MINIMIZER
			best_clf_precision = np.argmax(proba, axis=1)
			Y_pred_meta = np.array(list(map(lambda a: [1 if i == a else 0 for i in range(p)], \
											best_clf_precision)))
			Y_pred = predict_with_meta_dataset(ensemble, Y_pred_meta, X_test)
			if save:
				save_method_results(cv_path,
									method_name='precision_minimizer', 
									Y_pred=Y_pred, 
									Y_pred_meta=Y_pred_meta)
			if verbose:
				print("For PM: {0}".format(accuracy_score(Y_test, Y_pred)))

			# BINARY RELEVANCE
			Y_pred_meta = ml_clf.predict(X_test)
			Y_pred = predict_with_meta_dataset(ensemble, Y_pred_meta, X_test)
			if save:
				save_method_results(cv_path,
									method_name='binary_relevance',
									Y_pred=Y_pred, 
									Y_pred_meta=Y_pred_meta)
			if verbose:
				print("For BR: {0}".format(accuracy_score(Y_test, Y_pred)))

			# LABEL POWERSET
			Y_meta_val_lp = np.array(list(map(lambda row: "".join([str(r) for r in row]), Y_val_meta)))
			log_ovr = LogisticRegression(multi_class='ovr')
			log_ovr.fit(X_val, Y_meta_val_lp)
			Y_meta_test_lp = log_ovr.predict(X_test)
			Y_pred_meta = np.array([[int(i) for i in list(l)] for l in Y_meta_test_lp])
			Y_pred = predict_with_meta_dataset(ensemble, Y_pred_meta, X_test) 
			if save:
				save_method_results(cv_path,
									method_name='label_powerset',
									Y_pred=Y_pred, 
									Y_pred_meta=Y_pred_meta)
			if verbose:
				print("For LP: {0}".format(accuracy_score(Y_test, Y_pred)))

			# CLASSIFIER CHAIN
			clf = LogisticRegression()
			chain = train_chain(clf, X_val, Y_val_meta)
			Y_pred_meta = cc_predict(chain, X_test)
			Y_pred = predict_with_meta_dataset(ensemble, Y_pred_meta, X_test) 
			if save:
				save_method_results(cv_path,
									method_name='classifier_chain',  
									Y_pred=Y_pred, 
									Y_pred_meta=Y_pred_meta)
			if verbose:
				print("For CC: {0}".format(accuracy_score(Y_test, Y_pred)))

			# PROBABILISTIC CLASSIFIER CHAIN			   
			Y_pred_meta = []
			for id_test, x_test in enumerate(X_test):
				time1 = time.time()
				probs = []
				risks = []
				best_risk = np.inf
				if mc is None:
					# No Monte Carlo sampling
					patterns = list(itertools.product([0, 1], repeat=p))
					for pattern in patterns:
						prob = pcc_probability(chain, x_test, pattern)
						probs.append(prob)
					for solution in patterns:
						risk = sum([prob*true_loss(np.array(pattern), np.array(solution)) for pattern, prob in zip(patterns, probs)])
						risks.append(risk)
						if risk <= best_risk:
							best_solution = solution
							best_risk = risk
				else:
					mt_patterns = []
					for n_mt in range(mc):
						pattern = pcc_monte_carlo(chain, x_test, seed=n_mt)
						mt_patterns.append(pattern)
					#for solution in patterns: (monte carlo pas rigoureux...)
					for solution in mt_patterns:
						risk = sum([true_loss(np.array(pattern), np.array(solution)) for pattern in mt_patterns])
						risks.append(risk)
						if risk <= best_risk:
							best_solution = solution
							best_risk = risk
				time2 = time.time()
				if verbose:
					print("PPC, for test instance n {0}: {1}s".format(id_test, time2-time1))
				Y_pred_meta.append(list(best_solution))
			Y_pred_meta = np.array(Y_pred_meta)
			Y_pred = predict_with_meta_dataset(ensemble, Y_pred_meta, X_test)
			if save:
				save_method_results(cv_path,
									method_name='probabilistic_classifier_chain', 
									Y_pred=Y_pred, 
									Y_pred_meta=Y_pred_meta)
			if verbose:
				print("For PCC: {0} (time: {1}s)".format(accuracy_score(Y_test, Y_pred), time2-time1))

		print()


































