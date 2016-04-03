"""feature learning model training

This file contains the code for training a 
denoise autodecoder with noise rate of 0.3.

"""

from dA import dA

# allocate symbolic variables for the data
index = T.lscalar()    # index to a [mini]batch
x = T.matrix('x')  # the data

#####################################
# BUILDING THE MODEL CORRUPTION 30% #
#####################################

rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(
    numpy_rng=rng,
    theano_rng=theano_rng,
    input=x,
    #--- need to be changed to the number of features
    n_visible=num_of_features,
    #--- need to be changed to the dimension of hidden units
    n_hidden=num_of_hidden
)

cost, updates = da.get_cost_updates(
    corruption_level=0.3,
    #--- adjust the learning_rate
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
    # go through trainng set
    c = []
    for batch_index in range(n_train_batches):
        c.append(train_da(batch_index))

    print('Training epoch %d, cost ' % epoch, numpy.mean(c))

end_time = timeit.default_timer()

training_time = (end_time - start_time)

print(('The 30% corruption code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
