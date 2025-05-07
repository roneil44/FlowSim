######
# Main file for incompressible flow solver
# #######

## Imports
from vector import *

import numpy as np
from matplotlib import pyplot as plt


#### First generate square mesh and add it to the global variables
# Setup basic mesh parameters and declar them as global variables
#Initilize all global variables
x_max = 1
y_max = 1
number_x_points = 5
number_y_points = 4

## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points

dx = x_max / number_x_points
dy = y_max / number_y_points

## Boundary Wall velocities ##
top_wall = (1, 0) # u, v velocity
left_wall = (0, 0)
right_wall = (0, 0)
bottom_Wall = (0 , 0)

#Array initilizations
# Create as 2D arrays that get stacked into a single vector q
u_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
v_vel = np.zeros((nx*ny)) #Slightly larger than needed but makes iteration easier
pressures = np.zeros((nx, ny)) #Slightly larger, one pressure should be pinned

# Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

# Presure grid locations
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

#####
# Midterm compare vector operations

# First compute the gradient of a known function
# Using the pressure array since it already exists
for i in range(len(pressures)):
    for j in range(len(pressures[1])):
        pressures[i, j] = (2*i)+j
print(pressures)

pressure_gradient = gradient(pressures, dx, dy)
print(pressure_gradient)

plt.figure(1)
plt.contourf(X_pressures, Y_pressures, pressures)
plt.colorbar()
# plt.show()



##### PLOTTING FOR REPORTS #####

#Plotting staggered grid
#plt.figure(2)
plt.scatter(X_pressures, Y_pressures, c='black', marker='*')
plt.scatter(X_u, Y_u, c='blue', marker='>')
plt.scatter(X_v, Y_v, c='blue', marker='^')
plt.scatter(X_w, Y_w, c='red', marker='o')
plt.xticks(np.linspace(0,x_max, number_x_points+1))
plt.yticks(np.linspace(0,y_max,number_y_points+1))
plt.grid(True, alpha=.2)
plt.xlim(0,x_max)
plt.ylim(0, y_max)
plt.show()
