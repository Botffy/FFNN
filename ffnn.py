from itertools import chain


step_function = lambda x: -1 if x<0 else 1
identity = lambda x: x

class Perceptron:
	""" Perceptron model. """

	def __init__(self, weights, function):
		""" Initialize the perceptron with weights and an activation function. Weights[0] is the threshold value. """

		if(len(weights)<2):
			raise ValueError("The weight vector should have at least two elements")

		self.weights = weights
		self.function = function

	def evaluate(self, input):
		""" Calculate the output of the perceptron for a given input vector. """

		if len(input) != len(self.weights)-1:
			raise ValueError("Input vector is supposed to be one shorter than the perceptron's weight vector")

		return self.function(sum(x*w for x, w in zip(chain([-1], input), self.weights)))




class FFNN:
	""" Feed-forward neural network as a list of layers. """

	def __init__(self, layers):
		""" Create the network with a list of layers (lists of perceptrons) """

		self.layers = layers

	def evaluate(self, input):
		"""
		Calculate the output of the network for a given input vector.
		We just calculate the output of each neuron in each layer. The output is the output of the
		last layer
		"""

		for layer in self.layers:
			output = [ neuron.evaluate(input) for neuron in layer ]
			input = output

		return output

	def input_len(self):
		return len(self.layers[0][0].weights)-1

	def output_len(self):
		return len(self.layers[-1])
