"""
neural_network.py

The main module of the Neural Network from Scratch project, implementing a fully connected 
feedforward neural network using only NumPy. This module contains the core `NeuralNetwork` 
class, which supports customizable architectures, common activation functions, and 
gradient-based training via backpropagation.

Key Features:
- Customizable network architecture (input/hidden/output layers)
- Activation functions: ReLU, Sigmoid, Tanh
- Loss functions: Mean Squared Error (MSE), Cross-Entropy
- Batch gradient descent optimization
- He/Xavier weight initialization
- Modular design for easy extension

Classes:
    NeuralNetwork: Core class implementing network initialization, forward/backward passes, 
                  and training routines.

Usage:
    >>> from neural_network import NeuralNetwork
    >>> nn = NeuralNetwork(layers=[4, 5, 3], activation='relu', loss='cross_entropy')
    >>> nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)
    >>> predictions = nn.predict(X_test)

Dependencies:
    numpy (>= 1.19.0)

Author: Milan
Date: 21/04/2025
Version: 1.0
"""

# Import numpy
import numpy as np

class NeuralNetwork:
    def __init__(self, layers: list, activation='relu', output_activation=None, loss='mse', random_seed=None):
        """
        Initialization of the Neural Network

        Args:
            - layers (list): Takes a list of integers specifying the neurons per layer.
                Example: [input_size, hidden_size, ..., output_size]
            - activation (str): Activation function ('sigmoid', 'relu', or 'tanh')
            - output_activation (str): In case of multiclassification set the output_activation to 'softmax'
            - loss (str): Loss type ('mse' or 'cross_entropy')
            - random_seed (int): Sets the random seed for reproducability.
        """
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.weights = []
        self.bias = []

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize the weights and biases (He initialization for ReLU and Xavier for others)
        for i in range(len(layers) - 1):
            if activation == 'relu':
                scale = np.sqrt(2.0 / layers[i])
            else:
                scale = np.sqrt(1.0 / layers[i])

            self.weights.append(np.random.randn(layers[i], layers[i+1]) * scale)
            self.bias.append(np.zeros((1, layers[i+1])))


    def _activate(self, z, is_output=False):
        """ 
        Apply the activation functions

        Args:
            z (np.array): The weighted sum of inputs for a given layer, plus the bias term calculated in the forward
                method during the forward pass in the network.

        Options:
            - relu: Rectified Linear Unit (better known as ReLU) 
                https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
            - sigmoid: Sigmoid activation function:
                https://en.wikipedia.org/wiki/Sigmoid_function 
            - tanh: Hyperbolic tangent function:
                https://en.wikipedia.org/wiki/Hyperbolic_functions
        """
        if is_output and self.output_activation == 'softmax':
            exp = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)

    
    def _activate_derivative(self, z, is_output=False):
        """
        Calculates the derivative of the activation function. This will be used in the backward propagation
        part of the algorithm to adjust weights in a way that minimizes the loss function.

        Args: 
            z (np.array): The weighted sum of inputs for a given layer, plus the bias term calculated in the forward
                method during the forward pass in the network.
        """
        if is_output and self.output_activation == 'softmax':
            # Softmax derivative is handled directly in backprop (special case)
            return 1
        if self.activation == 'relu':
            return (z > 0).astype(float)
        if self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        if self.activation == 'tanh':
            return 1 - np.tanh(z)**2


    def _compute_loss(self, y_true, y_pred):
        """
        Calculates the loss function.

        Args:
            - y_true (np.array): An array containing the actual values of y
            - y_pred (np.array): An array containing the predicted values of y

        Options:
            - mse: Mean Squarred Error loss function
                https://en.wikipedia.org/wiki/Mean_squared_error
            - cross_entropy: Cross Entropy loss function
                https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning
        """
        if self.loss == 'mse':
            return np.mean((y_true - y_pred)**2)
        if self.loss == 'cross_entropy':
            # Clip to avoid log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred))


    def forward(self, X):
        """
        The forward propagation of the Neural Network involved the process where input data X is being
        passed forward through the layers to generate an output. 

        link: https://www.geeksforgeeks.org/what-is-forward-propagation-in-neural-networks/

        Args:
            - X (np.array): Set of input values for the neural network. 
        """
        self.layer_outputs = [X] # Store the outputs of each layer, including the input layer

        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.bias[i]
            # Pass is_output=True for the final layer
            a = self._activate(z, is_output=(i == len(self.weights)-1))
            self.layer_outputs.append(a)

        return self.layer_outputs[-1]


    def backward(self, y_true, learning_rate):
        """
        Backward propagation is the process of adjusting the weights and biases of the network
        in order to minimize the error or cost function.

        link: https://www.geeksforgeeks.org/backpropagation-in-neural-network/

        Args:
            - y_true (np.array): An array containing the actual values of y
            - learning_rate: The rate of which the neural network will adjust the weights and biases
        """
        gradients = []
        m = y_true.shape[0]  # Number of samples
        
        # Calculate output layer error
        if self.output_activation == 'softmax' and self.loss == 'cross_entropy':
            error = (self.layer_outputs[-1] - y_true) / m  
        elif self.loss == 'cross_entropy' and self.activation == 'sigmoid':
            error = (self.layer_outputs[-1] - y_true.reshape(-1,1)) / m
        else:
            if self.loss == 'mse':
                error = 2 * (self.layer_outputs[-1] - y_true.reshape(-1,1)) * self._activate_derivative(self.layer_outputs[-1], is_output=True) / m
            elif self.loss == 'cross_entropy':
                error = (self.layer_outputs[-1] - y_true.reshape(-1,1)) / m
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Gradient for weights and biases
            dw = np.dot(self.layer_outputs[i].T, error)
            db = np.sum(error, axis=0, keepdims=True)
            
            gradients.append((dw, db))

            # Propagate error to previous layer
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self._activate_derivative(self.layer_outputs[i])

        # Update weights and biases (reverse the gradients list to match layer order)
        gradients = gradients[::-1]
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.bias[i] -= learning_rate * gradients[i][1]


    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """
        Train the neural network using the forward and backward propagation to update the weights
        and biases of the neural network.

        Args: 
            - X (np.array): Set of input values for the neural network.
            - y (np.array): List of expected output/target variable
            - epochs (int): Total number of iterations in the training process
            - learning_rate (float): The learning rate of the algorithm
            - verbose (bool): True for printing the intermediate steps and False for not printing
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute the loss
            loss = self._compute_loss(y, y_pred)

            # Backward pass
            self.backward(y, learning_rate)
        
            if verbose and epoch % 100 == 0:
                print(f"Epoch: {epoch}, loss: {loss}")


    def predict(self, X):
        """
        Make the predictions using the neural network for a given input.

        Args: 
            - X (np.array): Set of input values for the neural network.
        """
        if self.layers[-1] == 1:
            return (self.forward(X) > 0.5).astype(int)
        else:
            return np.argmax(self.forward(X), axis=1)


    def predict_proba(self, X):
        """
        Return the probabilities of predictions made by the neural network.

        Args: 
            - X (np.array): Set of input values for the neural network.
        """
        return self.forward(X)