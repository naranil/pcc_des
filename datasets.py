import pandas as pd
import numpy as np

import os
import re
import operator
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import platform

datasets = {
			'adult' : 'adult/adult.data',

			'automoto': 'automoto/automoto.csv',

			'basehock': ['basehock/data.csv', 'basehock/target.csv'],

			'breast_cancer' : 'breast-cancer/breast-cancer-wisconsin.data',

			'colic' : 'colic/horse-colic.data',

			'colon' : ['colon/data.csv', 'colon/target.csv'],

			'credit_approval' : 'credit-approval/crx.data',

			'credit_german' : 'german/german.data-numeric',

			'elecrypt' : 'elecrypt/elecrypt.csv',

			'gunmid': 'gunmid/gunmid.csv',

			'heart_statlog' : 'heart/heart.dat',

			'ionosphere' : 'ionosphere/ionosphere.data',

			'krvskp': 'krvskp/krvskp.csv',

			'madelon' : ['madelon/madelon_train.data', 'madelon/madelon_train.labels'],

			'ovarian' : 'ovarian/ovarian.txt',

			'pcmac' : ['pcmac/data.csv', 'pcmac/target.csv'],

			'promoters' : 'promoters/promoters.data',

			'relathe': 'relathe/relathe.csv',

			'sonar' : 'sonar/Sonar.txt',

			'spambase' : 'spam/spambase.data,',

			'vote' : 'vote/house-votes-84.data',
			}
absolute_path = './datasets/'

def load_datasets(dataset_name):
	X, y = globals()['load_' + dataset_name]()

	return X, y

def load_adult():
	path = os.path.join(absolute_path, datasets['adult'])
	data = pd.read_csv(path, header=None)
	X = data.iloc[:, :14]
	y = data.iloc[:, 14]
	y = y.replace(' >50K', 1)
	y = y.replace(' <=50K', 0)
	X = X.replace(' ?', np.nan)
	X_new = X[~pd.isnull(X).any(axis=1)]
	y_new = y[~pd.isnull(X).any(axis=1)].astype(int)
	X_num = pd.DataFrame()
	for id_col in range(X_new.shape[1]):
		lb = LabelEncoder()
		X_num[id_col] = lb.fit_transform(X_new.iloc[:, id_col])
	return X_num, y_new

def load_automoto():
	path = os.path.join(absolute_path, datasets['automoto'])
	data = pd.read_csv(path)
	X = data.drop(['class'], axis=1)
	y = data['class']
	y = y - 8
	return X, y

def load_basehock():
	path = datasets['basehock']
	X = pd.read_csv(os.path.join(absolute_path, path[0]), header=None)
	y = pd.read_csv(os.path.join(absolute_path, path[1]), header=None)
	y = y.replace(1, 0)
	y = y.replace(2, 1)
	return X.astype(np.float), y[0]

def load_breast_cancer():
	path = os.path.join(absolute_path, datasets['breast_cancer'])
	data = pd.read_csv(path, header=None)
	data = data[data[6] != '?']
	X = data.ix[:, 1:9]
	y = data.ix[:, 10]
	y = 0.5 * y - 1
	return X.astype(np.float), y.astype(int)

def load_colon():
	path = datasets['colon']
	X = pd.read_csv(os.path.join(absolute_path, path[0]), header=None)
	y = pd.read_csv(os.path.join(absolute_path, path[1]), header=None)
	y -= 1
	y = y.values.ravel()
	return X, pd.Series(y)

def load_colic():
	path = os.path.join(absolute_path, datasets['colic'])
	data = pd.read_csv(path, header=None, sep=' *')
	for i in range(data.shape[1]):
		data.iloc[:, i] = data.iloc[:, i].replace('?', np.nan)
	data = data.fillna(data.mean(axis=1))
	X = data.iloc[:, :27]
	y = data.iloc[:, 27]
	y -= 1
	return X, y

def load_credit_approval():
	path = os.path.join(absolute_path, datasets['credit_approval'])
	data = pd.read_csv(path, header=None)
	data = data.replace('?', np.nan)
	data = data.dropna()

	y = data.iloc[:, 15]
	X = data.iloc[:, :15]

	y = y.replace('+', 1)
	y = y.replace('-', 0)
	# Easy binarization
	X.iloc[:, 0] = X.iloc[:, 0].replace('a', 0)
	X.iloc[:, 0] = X.iloc[:, 0].replace('b', 1)
	X.iloc[:, 8] = X.iloc[:, 8].replace('t', 0)
	X.iloc[:, 8] = X.iloc[:, 8].replace('f', 1)
	X.iloc[:, 9] = X.iloc[:, 9].replace('t', 0)
	X.iloc[:, 9] = X.iloc[:, 9].replace('f', 1)
	X.iloc[:, 11] = X.iloc[:, 11].replace('t', 0)
	X.iloc[:, 11] = X.iloc[:, 11].replace('f', 1)
	num_data = X.iloc[:, [0, 1, 2, 7, 8, 9, 10, 13, 14]].values

	# Non trivial binarization
	lb = LabelBinarizer()
	X3 = lb.fit_transform(X.iloc[:, 3])
	X4 = lb.fit_transform(X.iloc[:, 4])
	X5 = lb.fit_transform(X.iloc[:, 5])
	X6 = lb.fit_transform(X.iloc[:, 6])
	X12 = lb.fit_transform(X.iloc[:, 12])

	final_X = np.concatenate((num_data, X3, X4, X5, X6, X12), axis=1)

	return pd.DataFrame(final_X), y

def load_credit_german():
	path = os.path.join(absolute_path, datasets['credit_german'])
	data = pd.read_csv(path, header=None, sep=' *')
	X = data.iloc[:, :24]
	y = data.iloc[:, 24]
	y -= 1
	return X, y

def load_elecrypt():
	path = os.path.join(absolute_path, datasets['elecrypt'])
	data = pd.read_csv(path)
	X = data.drop(['class'], axis=1)
	y = data['class']
	y -= 12
	return X, y

def load_gunmid():
	path = os.path.join(absolute_path, datasets['gunmid'])
	data = pd.read_csv(path)
	X = data.drop(['class'], axis=1)
	y = data['class']
	y -= 17
	return X, y

def load_hepatitis():
	path = os.path.join(absolute_path, datasets['hepatitis'])
	data = pd.read_csv(path, header=None)
	data = data.replace('?', np.nan)
	data = data.fillna(data.mean(axis=1))
	X = data.iloc[:, 1:]
	y = data.iloc[:, 0]
	y -= 1
	return X, y

def load_ionosphere():
	path = os.path.join(absolute_path, datasets['ionosphere'])
	data = pd.read_csv(path, header=None)
	X = data.ix[:, :33]
	y = data.ix[:, 34]
	y = y.replace('g', 1)
	y = y.replace('b', 0)

	return X, y

def load_krvskp():
	path = os.path.join(absolute_path, datasets['krvskp'])
	data = pd.read_csv(path, header=None)
	X = data.iloc[:, :36]
	y = data.iloc[:, 36]
	y = y.replace('won', 1)
	y = y.replace('nowin', 0)
	X_num = pd.DataFrame()
	for id_col in range(X.shape[1]):
		lb = LabelEncoder()
		X_num[id_col] = lb.fit_transform(X.iloc[:, id_col])
	return X_num, y

def load_madelon():
	path = [os.path.join(absolute_path, p) for p in datasets['madelon']]
	X = pd.read_csv(path[0], sep=" ", header=None)
	y = pd.read_csv(path[1], header=None)
	y = y.replace(-1, 0)
	return X.iloc[:, :500].astype(np.float), y[0]

def load_ovarian():
	path = os.path.join(absolute_path, datasets['ovarian'])
	data = pd.read_csv(path, sep="\t", header=None)
	X = data.iloc[:, :1536]
	y = data[1536]
	y = y.replace(1, 0)
	y = y.replace(2, 1)
	return X.astype(np.float), y

def load_pcmac():
	path = [os.path.join(absolute_path, p) for p in datasets['pcmac']]
	X = pd.read_csv(path[0], header=None)
	y = pd.read_csv(path[1], header=None)
	y = y.replace(1, 0)
	y = y.replace(2, 1)
	return X.astype(np.float), y[0]

def load_relathe():
	path = os.path.join(absolute_path, datasets['relathe'])
	data = pd.read_csv(path, header=None)
	X = data.iloc[:, :4322]
	y = data.iloc[:, 4322]
	y -= 1
	return X, y

def load_sonar():
	path = os.path.join(absolute_path, datasets['sonar'])
	data = pd.read_csv(path, header=None, sep="\t")
	X = data.ix[:, :59]
	y = data.ix[:, 60]
	y -= 1
	return X, y

def load_spambase():
	path = os.path.join(absolute_path, datasets['spambase'])
	data = pd.read_csv(path, header=None)
	X = data.ix[:, :56]
	y = data.ix[:, 57]
	return X, y

def load_vote():
	path = os.path.join(absolute_path, datasets['vote'])
	data = pd.read_csv(path, header=None)
	data = data.replace('?', np.nan)
	data = data.dropna()

	y = data.ix[:, 0]
	X = data.ix[:, 1:]

	y = y.replace('republican', 1)
	y = y.replace('democrat', 0)
	X = X.replace('y', 1)
	X = X.replace('n', 0)

	return X, y
