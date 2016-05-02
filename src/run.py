import numpy
import pickle
import theano

from dA import dA
from train import test_dA
from NeuralNetwork.neural_network import test_mlp
from output import output_features
from data_loader import load_data
from data_loader import shared_dataset

path = "../data/processed/clean_synthesized_data.csv"
target_name = "hypertension"

datasets = load_data(path, target_name)
dA = test_dA(target_name=target_name, dataset=path)

features = output_features(path, target_name)
test_mlp(fileName=features, saveName="dA_model")