

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

		return self.function(sum(x*w for x, w in zip([-1]+input, self.weights)))

