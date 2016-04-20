"""data_loader.py
~~~~~~~~~~~~~~

A local data loader program that grab medical entries data from 
given csv file and randomly split data into a tuple of (train, validata, test)
Inside each group, data is represented in a tuple of (input, traget)

To make the loader more efficient, the program allows user to save 
and load the loader object through the use of pickle.

"""
import numpy
import pandas
import pickle

from theano import *
import theano.tensor as T

def read_csv(dataset):
    """ Reads an csv file and converts to (train, val, test)
    with each of the form (inputs, targets)

    :type dataset: string
    :param dataset: the path to the dataset

    """
    # default values
    target_name = "blood" # default target column
    train_percent = 0.5 # default trainset percentage
    val_percent = 0.2 # default valset percentage
    test_percent = 0.3 # default testset percentage

    # process dataset to form (inputs, targets)
    data = pandas.read_csv(dataset) # import from dataset
    targets = data[target_name].as_matrix() # get the target column as numpy array

    data = data.drop(target_name, 1) # get a dataframe with the remaining features
    inputs = data.as_matrix() # get the feature matrix as numpy matrix
    inputs = inputs[:, 1:] # drop the index column

    entries = targets.shape[0] # total entries of dataset

    # use permutation to generate random indices
    indices = numpy.random.permutation(entries) 

    # get randomized indices for each datasets
    train_idx = indices[:int(train_percent*entries)] 
    val_idx = indices[int(train_percent*entries):int(test_percent*entries)+int(train_percent*entries)]
    test_idx = indices[int(test_percent*entries)+int(train_percent*entries):]

    train = (inputs[train_idx,:], targets[train_idx])
    val = (inputs[val_idx,:], targets[val_idx])
    test = (inputs[test_idx,:], targets[test_idx])

    return train, val, test

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    
    """
    train_set, valid_set, test_set = read_csv(dataset)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def save_data(dataset, filename):
    data = load_data(dataset)
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(data, handle)

