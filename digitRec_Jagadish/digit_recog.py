from dnn_utils import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, layers_dims, params = None, learning_rate = 0.0007, mini_batch_size = 64, lambd=0, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 100, print_cost = True):
    """function that trains digit recognition model"""

    # L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    m = X.shape[1]                   # number of training examples
    
    if params is None:
        parameters = initialize_parameters(layers_dims)
        v, s = initialize_adam(parameters)
    else:
        parameters , v, s = params

    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        # seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = forward_propagation(minibatch_X, parameters, last_activation="softmax", rest_activation="relu")

            one_hot_Y = convert_to_one_hot(minibatch_Y,10)
            # Compute cost and add to the cost total
            cost_total += compute_cost(AL, one_hot_Y, params=parameters, lambd=lambd, average=True)

            # Backward propagation
            grads = backward_propagation(AL, one_hot_Y, caches,lambd=lambd, last_activation="softmax", rest_activation="relu")
            
            # print(gradient_check_n(parameters, grads, minibatch_X, minibatch_Y ))

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                            t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / len(minibatches)
        
        # Print the cost every 1000 epoch
        if print_cost and i % 5 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
            # predict(test_images, test_labels, parameters)
        if print_cost and i % 2 == 0:
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return (parameters, v, s)



# ================ loading and normalizing data ====================
# np.random.seed(1)
train_images, test_images, train_labels, test_labels = load_data()
train_images, test_images = train_images/255, test_images/255
train_labels, test_labels = train_labels.astype(int), test_labels.astype(int)

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)
# print(test_labels.dtype)

n = train_images.shape[0]
layers_dims = [n, 16, 16, 10]

# load_params = joblib.load('trained_model/digit_recog_model.pkl')
load_params = None

tic = time.time()
params = model(train_images,train_labels, params=load_params, layers_dims=layers_dims, learning_rate=0.01,
             mini_batch_size=512, lambd=0.7, num_epochs=21, print_cost=True)
toc = time.time()
print("time", toc-tic)

p1 = predict(train_images, train_labels, params[0])
p2 = predict(test_images, test_labels, params[0])
# joblib.dump(params, 'digit_recog_model.pkl')