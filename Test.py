# No Truce With The Furies
import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([2, 3])
        bias = 4

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = OurNeuralNetwork()
b = np.array([2, 3])
print(network.feedforward(b))
