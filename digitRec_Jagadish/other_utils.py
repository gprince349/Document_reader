import numpy as np

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        # keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    # parameters["W1"] = theta[:20].reshape((5,4))
    # parameters["b1"] = theta[20:25].reshape((5,1))
    # parameters["W2"] = theta[25:40].reshape((3,5))
    # parameters["b2"] = theta[40:43].reshape((3,1))
    # parameters["W3"] = theta[43:46].reshape((1,3))
    # parameters["b3"] = theta[46:47].reshape((1,1))

    parameters["W1"] = theta[:16*784].reshape((16,784))
    parameters["b1"] = theta[16*784:16*785].reshape((16,1))
    parameters["W2"] = theta[16*785:16*801].reshape((16,16))
    parameters["b2"] = theta[16*801:16*802].reshape((16,1))
    parameters["W3"] = theta[16*802:16*812].reshape((10,16))
    parameters["b3"] = theta[16*812:16*812+10].reshape((10,1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

import numpy as np

def gradient_check_n_test_case(): 
    np.random.seed(1)
    x = np.random.randn(4,3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5,4) 
    b1 = np.random.randn(5,1) 
    W2 = np.random.randn(3,5) 
    b2 = np.random.randn(3,1) 
    W3 = np.random.randn(1,3) 
    b3 = np.random.randn(1,1) 
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    
    return x, y, parameters