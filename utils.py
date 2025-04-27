"""
Utility functions for machine learning preprocessing and evaluation.

This module provides:
- StandardScaler: A class for standardizing features by removing mean and scaling to unit variance
- OneHotEncoder: A class for one-hot encoding categorical features
- train_test_split: A function to split data into random train and test subsets
- accuracy_score: A function to compute classification accuracy

Classes:
    StandardScaler: Feature standardization/normalization
    OneHotEncoder: One-hot encoding for categorical features

Functions:
    train_test_split(X, y, test_size, random_state) -> tuple
        Split arrays into random train and test subsets
    accuracy_score(y_true, y_pred) -> float
        Compute classification accuracy

Examples:
    >>> from utils import StandardScaler, OneHotEncoder, train_test_split, accuracy_score
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> encoder = OneHotEncoder()
    >>> y_encoded = encoder.fit_transform(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> acc = accuracy_score(y_true, y_pred)
"""

# Import numpy
import numpy as np

class StandardScaler:
    def __init__(self):
        """
        Initialization of the standard scaler for scaling the data.

        Args:
            - None
        """
        self.mean = None
        self.standard_deviation = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.

        Args: 
            - X (array): The data used to compute the mean and standard deviation used for later scaling along the features axis.
        """
        # Get the mean and standard deviation and store these
        self.mean = np.mean(X, axis=0)
        self.standard_deviation = np.std(X, axis=0)

        # Handle zero std (constant features)
        if np.any(self.standard_deviation == 0):
            self.standard_deviation[self.standard_deviation == 0] = 1.0

        return self


    def transform(self, X):
        """
        Transform the data.

        Args:
            - X (array): The data used to compute the mean and standard deviation used for later scaling along the features axis.
        """
        # Handle not fitted data
        if self.mean is None or self.standard_deviation is None:
            raise ValueError("The scaler must be fitted before transforming data.")

        return (X - self.mean) / self.standard_deviation


    def fit_transform(self, X):
        """
        Compute the mean and std and transform the data.

        Args: 
            - X (array): The data used to compute the mean and standard deviation used for later scaling along the features axis.
        """

        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self):
        """
        Initialize the OneHotEncoder

        Args:
            - None
        """
        self.mapping = {}
        self.reverse_mapping = {}

    def fit(self, data):
        """
        Fits the data to the mapping and stores the mapping result.

        Args:
            - data (array): takes a one dimensional array as input.
        """
        # Add the unique values in the data and assign the encoded value
        for i, name in enumerate(np.unique(data)):
            self.mapping[name] = i

        # Create the reversed mapping to decode the values
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

        return self

    def transform(self, data):
        """
        Transform the input data based on the mapping.

        Args: 
            - data (array): takes a one dimensional array as input.
        """
        # First encode the data
        encoded = np.vectorize(self.mapping.get)(data)

        # Apply the one hot encoding to get a vector of binary values
        one_hot_encoded = np.eye(len(self.mapping))[encoded]

        return one_hot_encoded

    def fit_transform(self, data):
        """
        Create the mapping dictionary and transform the array, returning the output.

        Args: 
            - data (array): takes a one dimensional array as input.
        """
        return self.fit(data).transform(data)

    def decode(self, data):
        """
        Decodes the output results of a model back to the original values.

        Args:
            - data (array): takes a one dimensional array as input.
        """
        return np.vectorize(self.reverse_mapping.get)(data)
        


def train_test_split(X: np.array,
                     y: np.array,
                     test_size=0.2,
                     random_state=None) -> np.array:
    """
    Split arrays or matrices into random train and test subsets.

    Args:
        - X (array): Contains the data to be features in the model.
        - y (array): Contains the target data.
        - test_size (float): The percentage of data to be used in the test set.
    """
    # Set the random seed if applicable
    if random_state is not None:
        np.random.seed(random_state)

    # Convert the input to np.array
    X = np.array(X)
    y = np.array(y)

    # Check whether the input shapes match
    if len(X) != len(y):
        raise ValueError("X and y must have the same numbers of samples")

    # Create the number of samples and test size
    n_samples = len(X) 
    n_test = int(len(X) * test_size)

    # Create shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Define the train and test indices
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]

    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
    

def accuracy_score(y_true: np.array,
                    y_pred: np.array) -> float:
    """
    Calculate the accuracy score as a metric to evaluate the model

    Args: 
        - y_true (array): The set of actual values.
        - y_pred (array): The set of predicted values by the model.
    """
    # Check if the two sets are of the same dimensions
    if len(y_true) != len(y_pred):
        raise ValueError("The two sets of values must be of same length.")

    # Make sure that the input data is 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Calculate the correct amount of predictions
    correct = np.sum(y_pred == y_true)
    total = len(y_true)

    return correct / total