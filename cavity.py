######
# Main file for incompressible flow solver
# #######



## Imports
from vector import *

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
from utils import *
from numpy import linalg


#### First generate square mesh and add it to the global variables
# Setup basic mesh parameters and declare them as global variables
#Initialize all global variables
x_max = 1
y_max = 1.2
number_x_points = 100
number_y_points = 100
dx = x_max / number_x_points
dy = y_max / number_y_points

## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points
nq = (nx-1)*ny + nx*(ny-1)
n_p = nx*ny-1


## Boundary Wall velocities ##
top_wall = (0, 0) # u, v velocity
left_wall = (0, 0)
right_wall = (0, 0)
bottom_Wall = (0 , 0)

#Array initializations
# Create as 2D arrays that get stacked into a single vector q
u_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
v_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
pressures = np.zeros((nx, ny)) #Slightly larger, one pressure should be pinned

# Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

# Pressure grid locations
X_pressures, Y_pressures = np.meshgrid(x_array, y_array, indexing='ij')

# U-Velocity grid locations
# Shift coords Right
x_array_u = x_array.copy()
x_array_u[:] = [i+dx/2 for i in x_array_u]
X_u, Y_u = np.meshgrid(x_array_u[:-1], y_array, indexing='ij')

# V-Velocity grid locations
# Shift coords Up
y_array_v = y_array.copy()
y_array_v[:] = [i+dy/2 for i in y_array_v]
X_v, Y_v = np.meshgrid(x_array, y_array_v[:-1], indexing='ij')

# Vorticity Locations for completeness
X_w, Y_w = np.meshgrid(x_array_u[:-1], y_array_v[:-1], indexing='ij')


######### Lid Driven Cavity Flow Solver #########

q = pack_q(u_vel, v_vel, nx, ny)
