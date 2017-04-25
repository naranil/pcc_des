import os
import pickle

def save_method_results(path, method_name=None, **kwargs):
	if method_name is not None:
		output_path = os.path.join(path, method_name)
		if not os.path.isdir(output_path):
			os.makedirs(output_path)
	else:
		output_path = path
	for data_name, data_value in kwargs.items():
		pickle.dump(data_value, open(os.path.join(output_path, data_name + '.pkl'), 'wb'))