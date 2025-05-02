########
# This script is used to produce a 2D rectangular staggered mesh for an inviscid flow solver
# Written by: Riley O'Neil
# Date: 4/29/25
# Course: MAE 250H
########


import numpy as np
from matplotlib import pyplot as plt


#Define boundaries
x_min = 0
x_max = 1
y_min = 0
y_max = 1

num_points_x = 15 
num_points_y = 15
#Assumes constant spacing
dx = (x_max-x_min) / num_points_x
dy = (y_max-y_min) / num_points_y

# See assignment but edges should be defined by velocity boundaries
#initialize arrays for meshgrid function
x_array = np.linspace(x_min, x_max, num_points_x)
y_array = np.linspace(y_min, y_max, num_points_y)

#remove Boundary cell centers for pressure centers
p_x_coord = x_array[1:-2]
p_y_coord = y_array[1:-2]
pressure_coordinates = np.meshgrid(p_x_coord, p_y_coord, indexing='ij')

# Shift array right one and remove 1 on the end for u-velocity coordinates
u_x_coord = x_array + [dx/2]
# Remove extra values
u_x_coord = u_x_coord[1:-3]
u_coordinates = np.meshgrid(u_x_coord, p_y_coord, indexing='ij')

# Shift array up one and remove extra components for v-velocity
v_y_coord = y_array + [dy/2]
v_y_coord = v_y_coord[1:-3]

v_coordinates = np.meshgrid(p_x_coord, v_y_coord, indexing='ij')

# USed both shifted for vorticity
w_coordinates = np.meshgrid(u_x_coord, v_y_coord, indexing='ij')




# Plot all coordinates
plt.figure(1)
plt.plot(pressure_coordinates[0], pressure_coordinates[1] ,marker ='*', color='k', linestyle='none')
plt.plot(u_coordinates[0], u_coordinates[1],marker ='>', color='b', linestyle='none')
plt.plot(v_coordinates[0], v_coordinates[1],marker ='^', color='b', linestyle='none')
plt.plot(w_coordinates[0], w_coordinates[1],marker ='o', color='r', linestyle='none')
plt.legend(['Pressure', 'U-Velocity', 'V-Velocity', 'Vorticity'])
plt.show() 



