from __future__ import division

import inspect
import collections
from itertools import chain
from itertools import product
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def step_function(x):
	return -1 if x<0 else 1

def identity(x):
	return x

def zero_step_function(x):
	return 0 if x<0 else 1


class Perceptron:
	""" Perceptron model. """

	def __init__(self, weights, function):
		""" Initialize the perceptron with weights and an activation function. Weights[0] is the threshold value. """

		if(len(weights)<2):
			raise ValueError("The weight vector should have at least two elements")

		self.weights = np.array(weights)
		self.function = function

	def evaluate(self, input):
		""" Calculate the output of the perceptron for a given input vector. """
		return self.function(np.dot(self.weights, np.array([-1] + list(input))))

	def __str__(self):
		return "w = {} b = {}".format(self.weights[1:], self.weights[0])




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
		input = np.array([-1] + list(input))

		for layer in self.layers:
			input = np.array([-1]+[ neuron.function(np.dot(neuron.weights, input)) for neuron in layer ])

		return input[1:]

	def input_len(self):
		return len(self.layers[0][0].weights)-1

	def output_len(self):
		return len(self.layers[-1])

	def create_graph(self):
		G = nx.DiGraph()
		for num, neuron in enumerate(self.layers[1]):
			G.add_weighted_edges_from( [((0,index-1), (1, num), weight) for index, weight in enumerate(neuron.weights) if weight != 0 and index!= 0 ] )
		for num, neuron in enumerate(self.layers[2]):
			G.add_weighted_edges_from( [((1,index-1), (2, num), weight) for index, weight in enumerate(neuron.weights) if index!= 0 ] )
		return G;

	def __str__(self):
		return "\n\n".join(["Layer {}:\n{}".format(index, "\n".join(map(str, [neuron for neuron in layer]))) for index, layer in enumerate(self.layers)])


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

	# first layer: for each input dimension, we have resolution-1 neurons, each with input_dim inputs
	first_layer = []
	for dimension in range(input_dim):
		input_weights = [1 if dim==dimension else 0 for dim in range(input_dim)]
		first_layer.extend( [Perceptron([domain[dimension][0]+num*units[dimension]] + input_weights, step_function) for num in range(1,resolution)] )

	second_layer = []
	xs = { }
	for square in product(range(resolution), repeat=input_dim):
		weights = [0]*len(first_layer)
		bias = -0.5
		xvalues = [None]*len(square)
		for dimension, area in enumerate(square):
			hb = area
			lb = area-1

			if lb >= 0:
				bias += 1
				weights[ (resolution-1)*(dimension) + lb ] = 1

			if hb < (resolution-1):
				bias += 1
				weights[ (resolution-1)*(dimension) + hb ] = -1

			midpoint = lb+0.5 if lb>=0 else hb-0.5
			xvalues[dimension] = domain[dimension][0] + (1+midpoint)*units[dimension]

		neuron = Perceptron([bias]+weights, zero_step_function)
		second_layer.append( neuron )
		xs[neuron] = xvalues

	third_layer = [
		Perceptron( [0] + [
			function(*xs[neuron]) if output_dim==1 else function(*xs[neuron])[outdim] for neuron in second_layer
		], identity ) for outdim in range(output_dim)
	]

	return FFNN([first_layer, second_layer, third_layer])
