import math
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import ffnn


def plot3d(func, domain, gridsize, plotres=50):
	print "Creating FFNN..."
	t0 = time.clock()
	approximator = ffnn.blumli(func, gridsize, domain)
	print "...done ({} sec)".format(time.clock()-t0)

	t0 = time.clock()
	print "Calculating {} values...".format(plotres**2)

	x = np.linspace(domain[0][0], domain[0][1], plotres)
	y = np.linspace(domain[1][0], domain[1][1], plotres)

	X, Y = np.meshgrid(x, y)
	#z = np.array( [func(a,b) for a, b in zip(np.ravel(X), np.ravel(Y))] )
	z_approx = np.array( [approximator.evaluate((a,b)) for a, b in zip(np.ravel(X), np.ravel(Y))] )
	#Z = z.reshape(X.shape)
	Z_approx = z_approx.reshape(X.shape)

	fig = plt.figure()
	ax = Axes3D(fig)
	#surf = ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.winter, rstride=1, cstride=1, linewidth=0, antialiased=False)
	surf = ax.plot_surface(X, Y, Z_approx, alpha=0.8, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False)
	#ax.set_zlim(-4, 3)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	print "...done ({} sec)".format(time.clock()-t0)
	print "Displaying..."
	plt.show()
	print "Done."
	return approximator

def draw(approximator):
	G = approximator.create_graph()

	maxlen = max([len(layer) for layer in approximator.layers])

	pos = {}
	for layer_idx, layer in enumerate(approximator.layers):
		for neuron_idx, neuron in enumerate(layer):
			pos[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx + maxlen - len(layer)/2)

	nx.draw_networkx_nodes(G, pos)
	nx.draw_networkx_edges(G, pos)
	plt.axis("off")
	plt.show()


if __name__ == "__main__":
	def func(x, y):
		return (math.cos(2.5*math.pi*x) + math.sin(1.5*math.pi*y)) * (x+y)

	domain = ((-1, 1),(-1, 1))

	approximator=plot3d(func, domain, gridsize=4)
	draw(approximator)
