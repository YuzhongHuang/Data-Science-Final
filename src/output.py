""" output.py
~~~~~~~~~~~~~~

This module will output the learned features given the 
original features with a trained dA

"""

import numpy
import pickle

from dA import dA
from data_loader import load_data


def get_features(da, inputs):
	features = []
	for inp in inputs:
		feature = da.get_reconstructed_input(da.get_hidden_values(inp)).eval()
		features.append(feature)
	return features

def output(da_name, inputs, save_name):
	da = dA.load(da_name)
	features = get_features(da, inputs[0:2])

	with open(save_name+'.pickle', 'wb') as handle:
        pickle.dump(features, handle)


datasets = load_data("../data/processed/clean_synthesized_data.csv")
train_set_x, train_set_y = datasets[0]

output("model", train_set_x.get_value(borrow=True))