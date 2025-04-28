# Neural Network from Scratch
### Overview
This project implements a fully connected neural network from scratch using only NumPy, without relying on high-level deep learning frameworks like TensorFlow or PyTorch. The goal is to understand the fundamental concepts of neural networks, including forward propagation, backpropagation, and gradient descent optimization.

### Features
- **Customizable Architecture:** Define the number of layers and neurons per layer.
- **Activation Functions:** Supports ReLU, Sigmoid, Tanh and Softmax activation functions.
- **Loss Functions:** Implements Mean Squared Error (MSE) and Cross-Entropy Loss.
- **Training Loop:** Includes forward pass, backpropagation, and weight updates.
- **Batch Gradient Descent:** Optimizes weights and biases using gradient descent.
- **Basic Data Preprocessing:** Includes normalization, splitting into training and testing sets and creating evaluation metrics.

### Dependencies
- Python 3.x
- NumPy

### Project Structure
```
neural_network_from_scratch/
│── neural_network.py          # Main neural network implementation
│── utils.py                   # Helper functions (normalization, encoding)
│── README.md                  # Project documentation
│── requirements.txt           # Dependencies
│── examples/                  # Example scripts
│   └── titanic_example.ipynb
│   └── iris_example.ipynb
```

### Future Improvements
- Add momentum and adaptive learning rate optimizers (e.g., Adam, RMSprop).
- Implement dropout and batch normalization.
- Support for convolutional layers (CNNs).
- Add visualization tools for loss tracking.

**Author:**
- **Name:** Milan
- **Github:** MilanKok98

Feel free to contribute or report issues
