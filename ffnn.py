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

	if input_dim == 1:
		domain = [(domain[0], domain[1])]

	units = [(domain[dim][1]-domain[dim][0])/resolution for dim in range(input_dim) ]

	# first layer: for each input dimensions, we have resolution-2 neurons, each with input_dim inputs
	first_layer = []
	for dimension in range(input_dim):
		input_weights = [1 if dim==dimension else 0 for dim in range(input_dim)]
		first_layer.extend(
			[Perceptron([domain[dimension][0]+num*units[dimension]] + input_weights, heaviside) for num in range(1,resolution)]
		)

	second_layer = [
		Perceptron(
			[0]*(len(first_layer)+1),
			step_function
		) for x in range( resolution**input_dim )
	]

	xs = { neuron: [None]*input_dim for neuron in second_layer }
	for index, neuron in enumerate(second_layer):
		neuron.weights[0] = -0.5
		for dimension in range(1,input_dim+1):
			#my bounds in given dimension
			lb = (index//(resolution**(dimension-1)) % resolution) + (resolution-1)*(dimension-1)
			hb = lb + 1

			if lb > (resolution-1)*(dimension-1):
				neuron.weights[0] +=1
				neuron.weights[lb] = 1
			else: lb = None
			if hb <= (resolution-1)*dimension:
				neuron.weights[0] +=1
				neuron.weights[hb] = -1
			else: hb = None

			unit = (domain[dimension-1][1]-domain[dimension-1][0])/resolution

			if lb is not None:
				xs[neuron][dimension-1] = domain[dimension-1][0] + (lb - (resolution-1)*(dimension-1) + 0.5)*unit
			else:
				xs[neuron][dimension-1] = domain[dimension-1][0] + (hb - (resolution-1)*(dimension-1) - 0.5)*unit

	third_layer = [
		Perceptron( [0] + [
			function(*xs[neuron]) if output_dim==1 else function(*xs[neuron])[outdim] for neuron in second_layer
		], identity ) for outdim in range(output_dim)
	]

	return FFNN([first_layer, second_layer, third_layer])
