#### Used to write wrapper functions to check the vector solvers
# These functions assume the vector fields used take the following forms:
# U = sin(m*pi*x/Lx)*sin(n*pi*y/Ly)
# V = sin(n*pi*x/Lx)*sin(m*pi*y/Ly)
# m and n will be given when calling the functions
# Pressure = 

### Imports ###
import numpy as np
from matplotlib import pyplot as plt
from vector import *
from utils import *

#Initialize all global variables
x_max = 1
y_max = 1
number_x_points = 10
number_y_points = 10
dx = x_max / number_x_points
dy = y_max / number_y_points

 ## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points
nq = (nx-1)*ny + nx*(ny-1)
n_p = nx*ny-1

# Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

# Presure grid locations
X_pressures, Y_pressures = np.meshgrid(x_array, y_array, indexing='ij')

# U-Velocity grid locations
# Shift coords Right
x_array_u = x_array.copy()
x_array_u[:] = [i-dx/2 for i in x_array_u]
X_u, Y_u = np.meshgrid(x_array_u, y_array, indexing='ij')

# V-Velocity grid locations
# Shift coords Up
y_array_v = y_array.copy()
y_array_v[:] = [i+dy/2 for i in y_array_v]
X_v, Y_v = np.meshgrid(x_array, y_array_v[:-1], indexing='ij')

# Vorticity Locations for completeness
X_w, Y_w = np.meshgrid(x_array_u[:-1], y_array_v[:-1], indexing='ij')

# Gradient check
u_p = np.zeros((nx, ny))

# #Linear 2*x gradient in X
# for i in range(nx):
#     for j in range(ny):
#         u_p[i,j] = i*dx*2

#Linear 2*x gradient in Y
for i in range(nx):
    for j in range(ny):
        u_p[i,j] = j*dy*2

u_g = gradient(u_p, dx, dy)

u_grad_x, u_grad_y = unpack_q(u_g, nx, ny)

plt.figure()
plt.contourf(X_pressures, Y_pressures, u_p)
plt.title('Pressuress')
plt.xlim((0,x_max))
plt.ylim((0,y_max))
plt.colorbar()

plt.figure()
plt.contourf(X_u, Y_u, u_grad_x)
plt.title('Pressure Gradient U')
plt.xlim((0,x_max))
plt.ylim((0,y_max))
plt.colorbar()

plt.figure()
plt.contourf(X_u, Y_u, u_grad_y)
plt.title('Pressure Gradient V')
plt.xlim((0,x_max))
plt.ylim((0,y_max))
plt.colorbar()

plt.show()
