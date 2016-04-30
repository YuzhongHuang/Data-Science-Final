""" output.py
~~~~~~~~~~~~~~

This module will output the learned features given the 
original features with a trained dA

"""

import numpy
import pickle
import theano

from dA import dA
from train import test_dA
from data_loader import load_data
from data_loader import shared_dataset


def get_features(da, inputs):
    """
    Given a set of features and an autoencoder
    generates a set of learned features from 
    the dataset

    """
    
    features = []
    for inp in inputs:
        feature = da.get_hidden_values(inp).eval()
        features.append(feature)
    return numpy.asarray(features)

def output(da, data_x, data_y):
    """
    Given an autoencoder and a set of features and target,
    generates a dataset in the form of theano shared data

    """

    features = get_features(da, data_x)
    new_data = (features, data_y)
    return shared_dataset(new_data)

def save(filename, pkl):
    """
    Save a file using pickle with specified filename

    """

    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(pkl, handle)

def output_features(path, target_name):
    """
    Given a file path of the dataset and the target name,
    returns a dataset of learned features from denoised
    autoencoder and save the dataset file 

    """

    datasets = load_data(path, target_name)
    dA = test_dA(target_name=target_name, dataset=path)

    train_set_x, train_set_y = datasets[0]
    train_set_x = train_set_x.get_value(borrow=True)
    train_set_y = train_set_y.eval()

    val_set_x, val_set_y = datasets[1]
    val_set_x = val_set_x.get_value(borrow=True)
    val_set_y = val_set_y.eval()

    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value(borrow=True)
    test_set_y = test_set_y.eval()

    print("\nTakes couple minutes, please be patient...\n")

    train = output(dA, train_set_x, train_set_y)
    val = output(dA, val_set_x, val_set_y)
    test = output(dA, test_set_x, test_set_y)

    output = (train, val, test)
    save("features_"+target_name, output)

    return output




