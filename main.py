from itertools import product

from ffnn import *


logical_and  = Perceptron([1.5, 1, 1], step_function)
logical_or   = Perceptron([-0.5, 1, 1], step_function)
logical_nand = Perceptron([-1.5, -1, -1], step_function)

logical_xor = FFNN([
	[logical_or, logical_nand],
	[logical_and]
])

for point in product([-1,1], repeat=2):
	print point, logical_xor.evaluate(point)

