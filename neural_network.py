# Import numpy
import numpy

class NeuralNetwork:
    def __init__(self, layers: list, activation:'relu', loss='mse'):
        """
        Initialization of the Neural Network

        Args:
            layers (list): Takes a list of integers specifying the neurons per layer.
                Example: [input_size, hidden_size, ..., output_size]
            activation (str): Activation function ('sigmoid', 'relu', or 'tanh')
            loss (str): Loss type ('mse' or 'cross_entropy')
        """
        self.layers = layers
        self.activation = activation
        self.loss = loss
        self.weights = []
        self.bias = []

        # Initialize the weights and biases (He initialization for ReLU and Xavier for others)
        for i in range(len(layers) - 1):
            if activation == 'relu':
                scale = np.sqrt(2.0 / layers[i])
            else:
                scale = np.sqrt(1.0 / layers[i])

            self.weights.append(np.random.randn())
            self.bias.append()

    def _activate(self, z):
        
        pass

    def _activate_derivative(self, z):

        pass

    def _compute_loss(self, y_true, y_pred):

        pass

    def forward(self, X):

        pass

    def backward(self, y_true, learning_rate):

        pass

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):

        pass

    def predict(self, X):

        pass
