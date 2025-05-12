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


#### First generate square mesh and add it to the global variables
# Setup basic mesh parameters and declar them as global variables
#Initilize all global variables
x_max = 1
y_max = 1.2
number_x_points = 25
number_y_points = 25
dx = x_max / number_x_points
dy = y_max / number_y_points

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
bottom_Wall = (0 , 0)

#Array initilizations
# Create as 2D arrays that get stacked into a single vector q
u_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
v_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
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
# for i in range(len(pressures)):
#     for j in range(len(pressures[0])):
#         pressures[i, j] = (2*i)+j
# #print(pressures)

# pressure_gradient = gradient(pressures, dx, dy)
# #print(pressure_gradient)

# plt.figure()
# plt.contourf(X_pressures, Y_pressures, pressures)
# plt.colorbar()
# # plt.show()


# Next compute divergence of known function
# for i in range(len(u_vel)):
#     for j in range(len(u_vel[0])):
#         u_vel[i,j] = j
#         v_vel[i,j] = 0
# divergences = div(u_vel, v_vel, dx, dy)
# print(u_vel)
# print(divergences)


# #v_vel = np.zeros((nx,ny))

# # Compute Laplacian of velocity arrays
# laplacian = lap(u_vel, v_vel, dx, dy)
# #print(laplacian)

# # Compute Non-linear advection
# advection = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_Wall)
# print(advection)


######## Analytic analysis of Vector operations #########

# Analytic comparison using sin functions
n = 2
m = 3

### SETUP Assign values to u and v arrays ###
#Since we slightly oversized our arrays u_vel iterators need to be slightly adjusted
for i in range(len(X_u)):
    for j in range(len(X_u[0])):
        u_vel[i+1,j] = math.sin(m*math.pi*X_u[i,j]/x_max)*math.sin(n*math.pi*Y_u[i,j]/y_max)

for i in range(len(X_v)):
    for j in range(len(X_v[0])):
        v_vel[i,j+1] = math.sin(n*math.pi*X_v[i,j]/x_max)*math.sin(m*math.pi*Y_v[i,j]/y_max)


### Divergence Check ####

# Check divergence
exact_divergence = np.zeros((nx, ny))

for i in range(len(X_pressures)):
    for j in range(len(X_pressures[0])):
        exact_divergence[i,j] = ((m*math.pi/x_max)*math.cos(m*math.pi*X_pressures[i,j]/x_max)*math.sin(n*math.pi*Y_pressures[i,j]/y_max) + 
        (m*math.pi/y_max)*math.sin(n*math.pi*X_pressures[i,j]/x_max)*math.cos(m*math.pi*Y_pressures[i,j]/y_max))

# Compute numeric divergence
numeric_divergence = div(u_vel, v_vel, dx, dy)

# Repack it into 2d array for plotting
num_div_array = np.zeros((nx, ny))
index = 0

for j in range(len(X_pressures[0])):
    for i in range(len(X_pressures)):
        if i != 0 or j != 0:
            num_div_array[i,j] = numeric_divergence[index]
            index += 1

# Calculate divergence error
div_error = np.absolute(np.subtract(exact_divergence, num_div_array))
div_error_sum = np.sum(div_error)
div_error_avg = div_error_sum/(len(div_error) * len(div_error[0]))
####

### Laplacian Check ####

# Initialize the two vector components of the laplacian
laplace_u = np.zeros((nx-1, ny))
laplace_v = np.zeros((nx, ny-1))

# print(np.shape(laplace_u))
# print(np.shape(X_u))

# print(np.shape(laplace_v))
# print(np.shape(X_v))
# print(Y_u[1,1])

## Solve exact for u
for i in range(len(laplace_u)):
    for j in range(len(laplace_u[0])):
        rh = math.sin(m*math.pi*(X_u[i,j])/x_max)*math.sin(n*math.pi*Y_u[i,j]/y_max)
        laplace_u[i,j] = -(m**2*math.pi**2/x_max**2)*rh - (n**2*math.pi**2/y_max**2)*rh
## Solve exact for v
for i in range(len(laplace_v)):
    for j in range(len(laplace_v[0])):
        rh = math.sin(n*math.pi*X_v[i,j]/x_max)*math.sin(m*math.pi*Y_v[i,j]/y_max)
        laplace_v[i,j] = -(n**2*math.pi**2/x_max**2)*rh - (m**2*math.pi**2/y_max**2)*rh

## Get numeric solution
num_lap = lap(u_vel, v_vel, dx, dy)

# Repack it into 2d array for plotting
num_lap_array_u = np.zeros((len(x_array_u)-1, len(y_array)))
num_lap_array_v = np.zeros((len(x_array), len(y_array_v)-1))
index = 0

# Build 2D U array
for j in range(len(num_lap_array_u[0])):
    for i in range(len(num_lap_array_u)):
        num_lap_array_u[i,j] = num_lap[index]
        index += 1

# Build 2D V array
for j in range(len(num_lap_array_v[0])):
    for i in range(len(num_lap_array_v)):
        num_lap_array_v[i,j] = num_lap[index]
        index += 1

# Co-locate U and V arrays on pressure centers
co_lap_u = collocate(laplace_u, X_u, Y_u, X_pressures, Y_pressures)
co_lap_v = collocate(laplace_v, X_v, Y_u, X_pressures, Y_pressures)

co_lap_u_num = collocate(num_lap_array_u, X_u, Y_u, X_pressures, Y_pressures)
vo_lap_v_num = collocate(num_lap_array_v, X_v, Y_v, X_pressures, Y_pressures)

print(num_lap_array_u)
print(laplace_u)

##### Plotting Midterm #####

# ### U Plot
# plt.figure()
# plt.contourf(X_u, Y_u, u_vel[1:])
# plt.title('U-velocity')
# plt.colorbar(label='Velocity')
# plt.xlim((0,x_max))
# plt.ylim((0,y_max))

# # V Plot
# plt.figure()
# plt.contourf(X_v, Y_v, [sublist[1:] for sublist in v_vel])
# plt.title('V-velocity')
# plt.colorbar(label='Velocity')
# plt.xlim((0,x_max))
# plt.ylim((0,y_max))

# # Divergence plot
# plt.figure()
# plt.contourf(X_pressures, Y_pressures, exact_divergence)
# plt.title('Exact Divergence')
# plt.colorbar(label='Divergence')
# plt.xlim((0,x_max))
# plt.ylim((0,y_max))

# #Numeric Divergence
# plt.figure()
# plt.contourf(X_pressures, Y_pressures, num_div_array)
# plt.title('Numeric Divergence')
# plt.colorbar(label='Divergence')
# plt.xlim((0,x_max))
# plt.ylim((0,y_max))

# #Divergence Error
# plt.figure()
# plt.contourf(X_pressures, Y_pressures, div_error, norm=matplotlib.colors.LogNorm())
# plt.title('Absolute Value of Divergence Error')
# plt.colorbar(label='Error')
# plt.annotate(f'dx = {dx}\ndy = {dy}\ntotal error = {round(div_error_sum, 3)}\navg point error = {round(div_error_avg, 5)}', (.1*x_max, .1*y_max))
# plt.xlim((0,x_max))
# plt.ylim((0,y_max))

# Laplacian U plots
plt.figure()
plt.contourf(X_u, Y_u, laplace_u)
plt.title('Exact Laplacian U-Component')
plt.colorbar(label='Laplace')
plt.xlim((0,x_max))
plt.ylim((0,y_max))

plt.figure()
plt.contourf(X_u, Y_u, num_lap_array_u)
plt.title('Numeric Laplacian U-Component')
plt.colorbar(label='Laplace')
plt.xlim((0,x_max))
plt.ylim((0,y_max))

# Laplacian V plots
plt.figure()
plt.contourf(X_v, Y_v, laplace_v)
plt.title('Exact Laplacian V-Component')
plt.colorbar(label='Laplace')
plt.xlim((0,x_max))
plt.ylim((0,y_max))

plt.figure()
plt.contourf(X_v, Y_v, num_lap_array_v)
plt.title('Numeric Laplacian V-Component')
plt.colorbar(label='Laplace')
plt.xlim((0,x_max))
plt.ylim((0,y_max))

print(np.shape(Y_pressures))
plt.figure()
plt.quiver(X_pressures, Y_pressures, co_lap_u, co_lap_v)
plt.title('Laplacian Quiver')
plt.colorbar(label='Laplace')
plt.xlim((0,x_max))
plt.ylim((0,y_max))

plt.show()


##### PLOTTING FOR REPORTS #####

####Plotting staggered grid######
# plt.figure()
# plt.title('Staggered Grid Formulation')
# plt.scatter(X_pressures, Y_pressures, c='black', marker='*')
# plt.scatter(X_u, Y_u, c='blue', marker='>')
# plt.scatter(X_v, Y_v, c='blue', marker='^')
# plt.scatter(X_w, Y_w, c='red', marker='o')
# plt.legend(['Pressure', 'U-velocity','V-velocity', 'Vorticity'],loc=1)
# plt.xticks(np.linspace(0,x_max, number_x_points+1))
# plt.yticks(np.linspace(0,y_max,number_y_points+1))
# plt.grid(True, alpha=.2)
# plt.xlim(0,x_max)
# plt.ylim(0, y_max)
# plt.show()
