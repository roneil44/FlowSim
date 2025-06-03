##### This script holds functions that assist in the development of an immersed boundary method

import numpy as np
import math
from immersed_boundary import *
from matplotlib import pyplot as plt

#Initialize all global variables
x_max = 1
y_max = 1
number_x_points = 50
number_y_points = 50
dx = x_max / number_x_points
dy = y_max / number_y_points

 ## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points
nq = (nx-1)*ny + nx*(ny-1)
n_p = nx*ny-1

##### Test Get circle coordinates function
x_c, y_c = get_points_on_circle(.25, nx, ny, dx, dy)

plt.figure()
plt.scatter(x_c, y_c)
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.show()