#######
# This file provides verification of different 
# types of advection equation solvers
# Written By: Riley O'Neil
# MAE250h
# 4/18/25
########

#######
# The purpose of this file is to solve the advection diffusion equation.
# It does this using a variety of different spatial and time discretization methods.
# The general form of the advection diffusion equation is:
#       (df/dt) + c*(df/dx) = 0


# Imports
import numpy as np
from matplotlib import pyplot as plt

# Constants
c = 1
grid_points = 100

# Grid generation
grid_locations = np.linspace(0,1, grid_points)

# Setup initial conditions
grid_values = np.zeros(grid_points)
grid_values[30:70] = 1

# Setup initialization
max_steps = 100
iteration = 1
error = 100
tolerance = 1e-12

time_step = .00001
total_time = .02
time = 0

#Upwind spatial, forward Euler time
# for i in range(len(grid_values)-1):
#     delta = grid_locations[i+1] - grid_locations[i]        
#     #Upwind discretization
#     print(grid_values[i+1] - (c*time_step/delta) * (-grid_values[i] + grid_values[i+1]))

# while time < total_time:
#     new_solution = []
    
#     # Upwind spatial, forward Euler time
#     for i in range(len(grid_values)-1):
#         delta = grid_locations[i+1] - grid_locations[i]        
#         #Upwind discretization
#         new_value = grid_values[i+1] - (c*time_step/delta) * (-grid_values[i] + grid_values[i+1])
#         #print(new_value)
#         new_solution.append(new_value)
        
    
#     # Central Difference Spatial, forward Euler time
#     # for i in range(len(grid_values)-2):
#     #     delta = grid_locations[i+1] - grid_locations[i]        
#     #     #Upwind discretization
#     #     #For first point use upwind discretization
#     #     if i == 0:
#     #         new_value = grid_values[i+1] - (c*time_step/delta) * (-grid_values[i] + grid_values[i+1])
        
#     #     else:
#     #         new_value = grid_values[i+1] - (c*time_step/2*delta) * (-grid_values[i] + grid_values[i+2])

#         #print(new_value)
#     #    new_solution.append(new_value)
    
    
#     # Replace with new solution
#     #print(new_solution)
#     grid_values[1::] = new_solution
    
#     time += time_step

# Forward Euler, spatial central differencing
while time < total_time:
    
    # Empty array for new solution
    new_solution = []
    
    for i in range(len(grid_values)):
        # Assumes constant spacing
        delta = grid_locations[1] - grid_locations[0]
        
        if i == 0:
            #When i is 0 use downstream values
            new_solution.append(grid_values[i] - (c*time_step/delta) * (-grid_values[i] + grid_values[i+1]))
        elif i == len(grid_values)-1:
            #When i is at last index use upstream values
            new_solution.append(grid_values[i] - (c*time_step/delta) * (-grid_values[i-1] + grid_values[i]))
        else:
            # Use central differencing for all other values
            new_solution.append(grid_values[i] - (c*time_step/(2*delta)) * (-grid_values[i-1] + grid_values[i+1]))
    
    #Update grid_values with new solution
    grid_values = new_solution
    
    time += time_step
    
    

# Plotting
plt.figure(1)
plt.plot(grid_locations, grid_values)    
plt.show()   




# Plotting
