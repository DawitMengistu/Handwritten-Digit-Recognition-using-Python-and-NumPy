import idx2numpy
import numpy as np
import json
# Load images (assuming you have already loaded images from the dataset)
images = idx2numpy.convert_from_file('./archive/t10k-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('./archive/t10k-labels.idx1-ubyte')


# Initialize the second layer with 16 neurons
n_inputs = 784  # Assuming input size from MNIST dataset
n_neurons_layer2 = 16
n_neurons_output = 10


lr = 0.06


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def calculateErr(result, aim):
    return 0.5 * ((aim - result) ** 2)


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = sigmoid(inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues * sigmoid_derivative(self.output)


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# layer1_weights.npy
# layer1_biases.npy


class Layer_Dense_Load:
    def __init__(self, n_inputs, n_neurons, pathWeight, pathBiases):
        self.weights = np.load(pathWeight)  # Load weights from file
        self.biases = np.load(pathBiases)    # Load biases from file
        assert self.weights.shape == (
            n_inputs, n_neurons), "Loaded weights shape does not match"
        assert self.biases.shape == (
            1, n_neurons), "Loaded biases shape does not match"

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


 # Training loop
# layer1 = Layer_Dense_Load(
#     n_inputs, n_neurons_layer2, "layer1_weights.npy", "layer1_biases.npy")
# layer2 = Layer_Dense_Load(
#     n_neurons_layer2, n_neurons_output, "layer2_weights.npy", "layer2_biases.npy")


# def numpy_to_list(arr):
#     return arr.tolist()


# # Save weights and biases for layer 1
# layer1_data = {
#     'weights': numpy_to_list(layer1.weights),
#     'biases': numpy_to_list(layer1.biases)
# }
# with open('data/layer1_data.json', 'w') as json_file:
#     json.dump(layer1_data, json_file)

# # Save weights and biases for layer 2
# layer2_data = {
#     'weights': numpy_to_list(layer2.weights),
#     'biases': numpy_to_list(layer2.biases)
# }
# with open('data/layer2_data.json', 'w') as json_file:
#     json.dump(layer2_data, json_file)


for i in range(5):
    epochs = 15
    errors = []
    batch_size = 100
    batch_accumulation = 100

    # layer1 = Layer_Dense(n_inputs, n_neurons_layer2)
    # layer2 = Layer_Dense(n_neurons_layer2, n_neurons_output)

    layer1 = Layer_Dense_Load(
        n_inputs, n_neurons_layer2, "layer1_weights.npy", "layer1_biases.npy")
    layer2 = Layer_Dense_Load(
        n_neurons_layer2, n_neurons_output, "layer2_weights.npy", "layer2_biases.npy")

    for epoch in range(epochs):
        epoch_error = 0
        accumulated_dW1 = 0
        accumulated_dB1 = 0
        accumulated_dW2 = 0
        accumulated_dB2 = 0

        for batch_idx in range(len(images) // batch_size):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Iterate over images in the batch
            for image_index in range(start_idx, end_idx):
                image = images[image_index]
                flattened_image = image.reshape(-1)

                # Forward pass
                layer1.forward(flattened_image)
                activation1 = Activation_Sigmoid()
                activation1.forward(layer1.output)

                layer2.forward(activation1.output)
                activation2 = Activation_Sigmoid()
                activation2.forward(layer2.output)

                outputLayer = activation2.output

                # Calculate output error
                target = np.eye(10)[labels[image_index]]
                error = calculateErr(outputLayer, target)
                total_error = np.sum(error)
                epoch_error += total_error

                # Backward pass for output layer
                dOut = -(target - outputLayer)
                activation2.backward(dOut)
                dNet2 = activation2.dinputs

                # Gradients for weights and biases between hidden layer and output layer
                dW2 = np.outer(activation1.output, dNet2)
                dB2 = dNet2

                # Calculate error term for hidden layer
                dA1 = np.dot(dNet2, layer2.weights.T)
                activation1.backward(dA1)
                dNet1 = activation1.dinputs

                # Gradients for weights and biases between input layer and hidden layer
                dW1 = np.outer(flattened_image, dNet1)
                dB1 = dNet1

                # Accumulate gradients
                accumulated_dW2 += dW2
                accumulated_dB2 += dB2
                accumulated_dW1 += dW1
                accumulated_dB1 += dB1

            # Update weights and biases after accumulating gradients over `batch_accumulation` batches
            if (batch_idx + 1) % batch_accumulation == 0:
                layer2.weights -= lr * (accumulated_dW2 / batch_accumulation)
                layer2.biases -= lr * (accumulated_dB2 / batch_accumulation)
                layer1.weights -= lr * (accumulated_dW1 / batch_accumulation)
                layer1.biases -= lr * (accumulated_dB1 / batch_accumulation)

                # Reset accumulated gradients
                accumulated_dW2 = 0
                accumulated_dB2 = 0
                accumulated_dW1 = 0
                accumulated_dB1 = 0

        # Calculate average epoch error
        average_epoch_error = epoch_error / len(images)
        errors.append(average_epoch_error)

        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Error: {average_epoch_error}')

    # Calculate percentage improvement
    # initial_error = errors[0]
    # final_error = errors[-1]
    # improvement = ((initial_error - final_error) / initial_error) * 100
    # print(f'Initial Error: {initial_error}')
    # print(f'Final Error: {final_error}')
    # print(f'Improvement: {improvement}%')

    # Save weights and biases for layer 1
    np.save('layer1_weights.npy', layer1.weights)
    np.save('layer1_biases.npy', layer1.biases)

    # Save weights and biases for layer 2
    np.save('layer2_weights.npy', layer2.weights)
    np.save('layer2_biases.npy', layer2.biases)


# print(outputLayer)
