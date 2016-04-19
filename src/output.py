""" output.py
~~~~~~~~~~~~~~

This module will output the learned features given the 
original features with a trained dA

"""

import numpy
import pickle

from dA import dA
from data_loader import load_data
from data_loader import shared_dataset


def get_features(da, inputs):
    features = []
    for inp in inputs:
        feature = da.get_reconstructed_input(da.get_hidden_values(inp)).eval()
        features.append(feature)
    return numpy.asarray(features)

def output(da_name, data_x, data_y, save_name):
    da = dA.load(da_name)
    features = get_features(da, data_x)
    new_data = (features, data_y)
    shared_new_data = shared_dataset(new_data)

    with open(save_name+'.pickle', 'wb') as handle:
        pickle.dump(shared_new_data, handle)

datasets = load_data("../data/processed/clean_synthesized_data.csv")
train_set_x, train_set_y = datasets[0]

train_set_x = train_set_x.get_value(borrow=True)
train_set_y = train_set_y.eval()

output("model", train_set_x, train_set_y, "features")