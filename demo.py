import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import ffnn


def func(x, y):
	return (math.cos(2.5*math.pi*x) + math.sin(1.5*math.pi*y)) * (x+y)

domain = {
	'x': (-1, 1),
	'y': (-1, 1)
}

# ask Blum-Li to creat an FFNN for func:
approximator = ffnn.blumli(func, 20, (domain['x'], domain['y']))


x = np.linspace(domain['x'][0], domain['x'][1], 30)
y = np.linspace(domain['y'][0], domain['y'][1], 30)

X, Y = np.meshgrid(x, y)
z = np.array( [func(a,b) for a, b in zip(np.ravel(X), np.ravel(Y))] )
z_approx = np.array( [approximator.evaluate((a,b)) for a, b in zip(np.ravel(X), np.ravel(Y))] )
Z = z.reshape(X.shape)
Z_approx = z_approx.reshape(X.shape)

fig = plt.figure()
ax = Axes3D(fig)
#surf = ax.plot_surface(X, Y, Z, alpha=0.3, cmap=cm.winter, rstride=1, cstride=1, linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z_approx, alpha=0.8, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
