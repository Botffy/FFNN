from itertools import product

from ffnn import *


#logical_and  = Perceptron([1.5, 1, 1], Heaviside)
#logical_or   = Perceptron([-0.5, 1, 1], Heaviside)
#logical_nand = Perceptron([-1.5, -1, -1], Heaviside)

#logical_xor = FFNN([
#	[logical_or, logical_nand],
#	[logical_and]
#])

#for point in product([-1,1], repeat=2):
#	print point, logical_xor.evaluate(point)

net = blumli(lambda x, y: (x,y), 4, [(-2,2),(-1,1)])


print net.evaluate([1,1])
