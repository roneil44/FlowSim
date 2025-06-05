##### This script holds functions that assist in the development of an immersed boundary method

import numpy as np
import math
from immersed_boundary import *
from matplotlib import pyplot as plt

#Initialize all global variables
x_max = 1
y_max = 1
number_x_points = 20
number_y_points = 20
dx = x_max / number_x_points
dy = y_max / number_y_points
ds = (dx+dy)/2

 ## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points
nq = (nx-1)*ny + nx*(ny-1)
n_p = nx*ny-1

# Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

x_array_u = x_array.copy()
x_array_u[:] = [i+dx/2 for i in x_array_u]
X_u, Y_u = np.meshgrid(x_array_u[:-1], y_array, indexing='ij')

##### Test Get circle coordinates function
x_c, y_c = get_points_on_circle(.25, nx, ny, dx, dy, ds)

## Calulate relationship between lagragian and cartesian points
cart_x, cart_y = calc_influence(X_u.flatten(), Y_u.flatten(), x_c, y_c, ds)
x_points = [tup[0] for tup in cart_x]
y_points = [tup[0] for tup in cart_y]

plt.figure()
plt.scatter(x_c, y_c)
plt.scatter(x_points, y_points, color='red')
plt.title('Immersed Boundary Region of Influence')
plt.legend(['Lagragian Boundary Points','Cartesian points with 2.5*ds'])
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.show()