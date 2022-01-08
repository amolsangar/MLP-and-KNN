# Machine Learning - MLP and KNN
Scratch implementation of Multilayer Perceptron (MLP) and K-Nearest Neighbor (KNN) algorithms

# KNN Classification
- The fit method for this function is implemented by simply mapping the input and output data to the class variables.

- In the predict method, there are two loops implemented thus running time complexity for this algorithm is O(n^2).

- For every data point in the test set, the distance to every other point in the training set is calculated by either Euclidean distance or Manhattan distance. After this, the distance array (result) is sorted in ascending order and voting is done based on uniform or distance weights for the K nearest neighbors.

- The algorithm consistently works well when compared with the output of sklearn models.

# MLP Classification
**Fit method** 
- It initializes random values to weights and biases for the output and the hidden layer. After this, the training starts and it uses these weights to perform forward propagation. 

- The linear function is used to calculate the dot product of the matrices along with adding biases. 

- Each layer output is passed through the activation function and the losses is calculated using the cross-entropy loss function and stored every 20 iterations. 

**Backpropagation** 
- The next step is to calculate the delta i.e., the difference between actual and predicted output and multiply with the derivative of the activation function. 

- To calculate the gradient, the input to the layer is multiplied with the delta obtained in the above step.

- The gradient is standardized to avoid the gradient explosion and clipped between -1 to 1.

- The last step is to then update the weights.

- These same steps are implemented for every layer.

- Once forward-propagation and backpropagation are completed, it is repeated again for the number of iterations provided.

**Predict** 
- The predict function only performs the forward-propagation and returns the result.

**Challenges** 
- ```RuntimeWarning: overflow encountered in exp```

  This warning comes when sigmoid and tanh activation functions are invoked and is caused due to the ‘np.exp’ value range. 

  Because of this, another error occurs which is - 
- ```RuntimeWarning: invalid value encountered in true_divide```

This results in poor performance at some of the inputs where tanh is used. 

To solve this, the values for tanh and sigmoid functions were clipped at +-500.


