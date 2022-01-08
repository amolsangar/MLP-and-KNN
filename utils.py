# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [AMOL DATTATRAY SANGAR] -- [asangar]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    dist = np.linalg.norm(x1-x2)
    return dist

def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    dist = np.abs(x2 - x1).sum()
    return dist

def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if(derivative):
        return 1
    else:
        return x

def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip( x, -500, 500 )
    f_x = 1/(1 + np.exp(-x))
    
    # Derivative
    f2_x = f_x * (1 - f_x)

    if(derivative):
        return f2_x
    else:
        return f_x

def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip( x, -500, 500 )
    f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    # Derivative
    f2_x = 1 - f_x**2

    if(derivative):
        return f2_x
    else:
        return f_x

def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if(derivative):
        return np.greater(x, 0).astype(int)
    else:
        return x * (x > 0)

def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    # prevent taking the log of 0
    eps = np.finfo(float).eps
    cross_entropy = -np.sum(y * np.log(p + eps))
    return cross_entropy

# Reference - https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
def one_hot_encoding(p):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """

    shape = (p.size, p.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(p.size)
    one_hot[rows, p] = 1

    return one_hot