import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Single-layer Perceptron
class SingleLayerPerceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size, 1)
        self.bias = np.random.rand(1)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for _ in range(epochs):
            # Forward pass
            output = self.predict(X)
            activation = sigmoid(output)

            # Error
            error = y - activation

            # Backward pass
            d_weights = np.dot(X.T, error * sigmoid_derivative(activation))
            d_bias = np.sum(error * sigmoid_derivative(activation))

            # Update weights and bias
            self.weights += learning_rate * d_weights
            self.bias += learning_rate * d_bias

# Multi-layer Perceptron
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    def predict(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_activation = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_activation, self.weights_hidden_output) + self.bias_output
        output = sigmoid(self.output_layer_input)

        return output

    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for _ in range(epochs):
            # Forward pass
            self.predict(X)

            # Compute error
            output_error = y - self.output_layer_input
            output_delta = output_error * sigmoid_derivative(sigmoid(self.output_layer_input))

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(sigmoid(self.hidden_layer_input))

            # Update weights and biases
            self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer_activation.T, output_delta)
            self.bias_output += learning_rate * np.sum(output_delta, axis=0)

            self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)
            self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Single-Layer Perceptron
print("Single-Layer Perceptron:")
slp = SingleLayerPerceptron(input_size=2)
slp.train(X, y)
predictions_slp = slp.predict(X)
print("Predictions:", np.round(sigmoid(predictions_slp), 2))

# Multi-Layer Perceptron
print("\nMulti-Layer Perceptron:")
mlp = MultiLayerPerceptron(input_size=2, hidden_size=4, output_size=1)
mlp.train(X, y)
predictions_mlp = mlp.predict(X)
print("Predictions:", np.round(predictions_mlp, 2))