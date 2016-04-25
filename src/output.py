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
        feature = da.get_hidden_values(inp).eval()
        features.append(feature)
        print len(features)
    return numpy.asarray(features)

def output(da_name, data_x, data_y):
    da = dA.load(da_name)
    features = get_features(da, data_x)
    new_data = (features, data_y)
    return shared_dataset(new_data)


def save(filename, pkl):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(pkl, handle)

datasets = load_data("../data/processed/clean_synthesized_data.csv")

train_set_x, train_set_y = datasets[0]
train_set_x = train_set_x.get_value(borrow=True)
train_set_y = train_set_y.eval()

val_set_x, val_set_y = datasets[1]
val_set_x = val_set_x.get_value(borrow=True)
val_set_y = val_set_y.eval()

test_set_x, test_set_y = datasets[2]
test_set_x = test_set_x.get_value(borrow=True)
test_set_y = test_set_y.eval()

train = output("model", train_set_x, train_set_y)
val = output("model", val_set_x, val_set_y)
test = output("model", test_set_x, test_set_y)

save("features", (train, val, test))
