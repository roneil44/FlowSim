######
# Main file for incompressible flow solver
# #######



## Imports
from vector import *

import numpy as np
from matplotlib import pyplot as plt
from utils import *
from numpy import linalg
from conjugant_solver import *


#### First generate square mesh and add it to the global variables
# Setup basic mesh parameters and declare them as global variables
#Initialize all global variables
x_max = 1
y_max = 1.2
number_x_points = 10
number_y_points = 10
dx = x_max / number_x_points
dy = y_max / number_y_points

v = 1

# Solver Settings
total_time = .2
dt = .01
# Tolerances
tol1 = 1e-3
tol2 = 1e-3


## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points
nq = (nx-1)*ny + nx*(ny-1)
n_p = nx*ny-1


## Boundary Wall velocities ##
top_wall = (1, 0) # u, v velocity
left_wall = (0, 0)
right_wall = (0, 0)
bottom_wall = (0 , 0)

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

# Grid corners for plotting collocated results
x_corners = np.linspace(0, x_max, nx+1)
y_corners = np.linspace(0, y_max, ny+1)
X_corners, Y_corners = np.meshgrid(x_corners, y_corners, indexing='ij')

######### Lid Driven Cavity Flow Solver #########
t = 0

# First pass initilizations
A_old = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_wall)

t += dt
# Setup while loop to iterate over time


while t <= total_time:

    q = pack_q(u_vel, v_vel, nx, ny)
    p = pack_p(pressures, nx, ny)

    ##### First Fractional Step
    # First calulate the Right hand side of the first fractional step
    A_new = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_wall)
    s = S_times(u_vel, v_vel, dt, v, nx, ny, dx, dy)
    #laplace = lap(u_vel, v_vel, dx, dy)
    
    # Partial RHS terms
    bc_laplace = dt*v*bc_lap(nx, ny, dx, dy, top_wall, left_wall, right_wall, bottom_wall)
    full_advect = (dt/2)*(np.add(np.multiply(3,A_new), A_old))

    # Full RHS
    RHS = np.add(s, full_advect)
    RHS = np.add(RHS, bc_laplace)

    u_F = conjugant_solve1(q, RHS, tol1, dt, v, nx, ny, dx, dy)
    
    ###### Second Fractional Step
    u_F_u, u_F_v = unpack_q(u_F, nx, ny)

    divergence = (1/dt)*div(u_F_u, u_F_v, dx, dy)
    divergence_boundary = (1/dt)*bc_div(nx, ny, dx, dy, top_wall, left_wall, right_wall, bottom_wall)
    
    RHS2 = np.add(divergence, divergence_boundary)
    
    p_plus = conjugant_solve2(p, RHS2, tol2, dt, v, nx, ny, dx, dy)

    ###### Third Step solve
    # Pressure term
    pressures = unpack_p(p_plus, nx, ny)
    press_grad = gradient(pressures, dx, dy)
    u_grad, v_grad = unpack_q(press_grad, nx, ny)

    # R inverse term
    first = (dt*v/2)*lap(u_grad, v_grad, dx, dy)
    second = np.power(first, 2)
    R_inverse = np.add(press_grad, first)
    R_inverse = np.add(R_inverse, second)

    R_inverse = dt*R_inverse

    u_new = np.subtract(u_F, R_inverse)
    
    # Repopulate arrays
    u_vel, v_vel = unpack_q(u_new, nx, ny)

    # Increment timestep
    t += dt


# # Temp for troubleshooting
#u_vel, v_vel = unpack_q(u_F, nx, ny)
print(u_vel, v_vel)
# Collocate velocities
U, V = collocate_velocity(u_vel, v_vel, nx, ny, top_wall, bottom_wall, right_wall, left_wall)


plt.figure()
plt.quiver(X_corners, Y_corners, U, V)
plt.xlim(0,x_max)
plt.ylim(0,y_max)

plt.figure()
plt.contourf(X_corners, Y_corners, U)
plt.colorbar()
plt.title("U Contour")

plt.figure()
plt.contourf(X_corners, Y_corners, V)
plt.colorbar()
plt.title("V contour")


plt.show()
