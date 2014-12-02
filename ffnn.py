from __future__ import division

import inspect
import collections
from itertools import chain


heaviside = lambda x: -1 if x<0 else 1
identity = lambda x: x
step_function = lambda x: 0 if x<0 else 1


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


def blumli(function, resolution, domain):
	"""
	Constructs an FFNN that approximates given numeric function.
	"""
	# fugly hacks galore to determine dimensions
	input_dim = len(inspect.getargspec(function).args)
	outsample = function( *range(input_dim))
	output_dim = len(list( outsample )) if isinstance(outsample, collections.Iterable) else 1

	if input_dim > 1:
		raise NotImplementedError("Sorry, only input dim 1 for now")

	unit = (domain[1]-domain[0]) / resolution

	# first layer: resolution-2 neurons, each with input_dim inputs
	first_layer = [
 		Perceptron( [domain[0]+num*unit, 1], heaviside) for num in range(1,resolution)
	]

	h = {0: 1, 1: -1}
	# second layer:
	second_layer = [
		Perceptron(
			[1.5 if x!=0 and x!=resolution-1 else 0.5]+[h.get(index+1-x, 0) for index, neuron in enumerate(first_layer)],
			step_function
		) for x in range(resolution)
	]

	third_layer = [
		Perceptron( [0] + [ function(domain[0]+(index+0.5)*unit)[outdim] for index, neuron in enumerate(second_layer) ], identity ) for outdim in range(output_dim)
	]

	return FFNN([first_layer, second_layer, third_layer])

