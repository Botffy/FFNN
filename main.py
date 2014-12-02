from ffnn import *


logical_and = Perceptron([1.5, 1, 1], step_function)
print logical_and.evaluate([1,1])
