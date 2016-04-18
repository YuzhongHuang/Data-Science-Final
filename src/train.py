"""train.py
~~~~~~~~~~~~~~

This file contains the code for training a 
denoise autodecoder with noise rate of 0.3.

"""

import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from data_loader import load_data

def test_dA(learning_rate=0.01, 
            training_epochs=100,
            dataset="../data/processed/clean_synthesized_data.csv",
            batch_size=50):

    """
    Feed data into denoise autoencoder to train a model,
    and test the model over test dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        inputs=x,
        # -- needs to declare the total input feature numbers
        n_visible=train_set_x.get_value(borrow=True)[0].shape[0],
        # -- needs to declare the total hidden unit feature numbers
        n_hidden=22
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        c = []
        # go through trainng set
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print 'Training epoch %d, cost: ' %epoch, numpy.mean(c)

    da.save("model")

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

if __name__ == '__main__':
    test_dA()