# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [AMOL DATTATRAY SANGAR] -- [asangar]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

     # Function to sort the list by first item of tuple
    def sortTuple(self,tup,rev=False): 
        return(sorted(tup, key = lambda x: x[0], reverse=rev))
    
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        final_result = []
        for i in X:
            result = []
            for j in range(0,len(self._X)):
                result.append((self._distance(i,self._X[j]), self._y[j]))
                        
            result = self.sortTuple(result,False)   # Sort by Distance

            if self.weights == "uniform":
                count = {}
                for n in range(0,self.n_neighbors):
                    count[result[n][1]] = count.get(result[n][1],0) + 1
                    
                count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))

                for c in count:
                    final_result.append(c)
                    break
            
            elif self.weights == "distance":
                weighted_count = {}
                for n in range(0,self.n_neighbors):
                    if(result[n][0] == 0):
                        weighted_count[result[n][1]] = weighted_count.get(result[n][1],0) + (1 / 0.01)  # if distance 0 then divide by 0.01
                    else:    
                        weighted_count[result[n][1]] = weighted_count.get(result[n][1],0) + (1 / result[n][0])
                
                weighted_count = dict(sorted(weighted_count.items(), key=lambda item: item[1], reverse=True))

                for w in weighted_count:
                    final_result.append(w)
                    break
                
        return final_result