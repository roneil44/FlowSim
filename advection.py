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
intial_values = np.zeros(grid_points)
lower = int(.3*grid_points)
upper = int(.7*grid_points)
intial_values[lower:upper] = 1

# Setup initialization
max_steps = 100
iteration = 1
error = 100
tolerance = 1e-12

time_step1 = .001
total_time1 = .5
time = 0

#Upwind spatial, forward Euler time
solution1 = intial_values.copy()

while time <= total_time1:
    new_solution = []
    
    # Upwind spatial, forward Euler time
    for i in range(len(solution1)-1):
        delta = grid_locations[i+1] - grid_locations[i]        
        #Upwind discretization
        new_value = solution1[i+1] - (c*time_step1/delta) * (-solution1[i] + solution1[i+1])
        #print(new_value)
        new_solution.append(new_value)
    
    # Replace with new solution
    #print(new_solution)
    solution1[1::] = new_solution

    # Store intermediate solution
    if time == time_step1*(total_time1/(2*time_step1)):
        solution1_5 = solution1.copy()

    time += time_step1
    time = round(time, len(str(time_step1))-2)

# Forward Euler, spatial central differencing
solution2 = intial_values.copy()

# reset time variables
time_step2 = .0001
total_time2 = .04
time = 0

while time <= total_time2:
    
    # Empty array for new solution
    new_solution = []
    
    for i in range(len(solution2)):
        # Assumes constant spacing
        delta = grid_locations[1] - grid_locations[0]
        
        if i == 0:
            #When i is 0 use downstream values
            new_solution.append(solution2[i] - (c*time_step2/delta) * (-solution2[i] + solution2[i+1]))
        elif i == len(solution2)-1:
            #When i is at last index use upstream values
            new_solution.append(solution2[i] - (c*time_step2/delta) * (-solution2[i-1] + solution2[i]))
        else:
            # Use central differencing for all other values
            new_solution.append(solution2[i] - (c*time_step2/(2*delta)) * (-solution2[i-1] + solution2[i+1]))
    
    #Update grid_values with new solution
    solution2 = new_solution
    
    # Store intermediate solution
    if time == total_time2/2:
        solution2_5 = solution2[:]

    time += time_step2
    time = round(time, len(str(time_step2))-2)


# Forward Euler, spatial downwind differencing
solution3 = intial_values.copy()

# reset time variables
time_step3 = .001
total_time3 = .01
time = 0

while time <= total_time3:
    
    # Empty array for new solution
    new_solution = []
    
    for i in range(len(solution3)):
        # Assumes constant grid spacing
        delta = grid_locations[1] - grid_locations[0]
        
        if i == len(solution3)-1:
            #When i is at last index use upstream values
            new_solution.append(solution3[i] - (c*time_step3/delta) * (-solution3[i-1] + solution3[i]))
        else:
            #Downstream calc
            new_solution.append(solution3[i] - (c*time_step3/delta) * (-solution3[i] + solution3[i+1]))
    
    #Update grid_values with new solution
    solution3 = new_solution

    # Store intermediate solution
    if time == total_time3/2:
        solution3_5 = solution3[:]
    
    time += time_step3
    time = round(time, len(str(time_step3))-2)


# Backward Euler, spatial upwinding
solution4 = intial_values.copy()

# reset time variables
time_step4 = .001
total_time4 = .5
time = 0



# while time <= total_time3:
    
#     # Empty array for new solution
#     new_solution = []
    
#     for i in range(len(solution4)):
#         # Assumes constant grid spacing
#         delta = grid_locations[1] - grid_locations[0]

#         # Implicit solver parameters
#         tol = 1e-12
#         iteration = 0
        
#         if i == 0:
#             #When i is at first index use downstream values
#             # Generate prediction for next time step
#             while True:
#                 if iteration == 0:
#                     # First guess use forward euler
#                     guessed_solution = solution4[i] - (c*time_step4/delta) * (-solution4[i-1] + solution4[i])
#                 else:
#                     # Otherwise use predictor corrector
#                     guessed_solution = guessed_solution - (c*time_step4/delta) * (-solution4[i] + solution4[i])

#             new_solution.append(solution4[i] - (c*time_step4/delta) * (-solution4[i] + solution4[i+1]))
#         else:
#             #Downstream calc
#             new_solution.append(solution4[i] - (c*time_step4/delta) * (-solution4[i] + solution4[i+1]))
    
#     #Update grid_values with new solution
#     solution4 = new_solution

#     # Store intermediate solution
#     if time == total_time3/2:
#         solution4_5 = solution4[:]
    
#     time += time_step4
#     time = round(time, len(str(time_step4))-2)

    

# Plotting
plt.figure(1)
plt.plot(grid_locations, intial_values)
plt.plot(grid_locations, solution1_5)
plt.plot(grid_locations, solution1)
plt.legend(['Initial Values', f'{total_time1/2} secs, dt {time_step1}', f'{total_time1} secs, dt {time_step1}'])
plt.title("Upstream Differencing")

plt.figure(2)
plt.plot(grid_locations, intial_values)
plt.plot(grid_locations, solution2_5)
plt.plot(grid_locations, solution2)
plt.legend(['Initial Values', f'{total_time2/2} secs, dt {time_step2}', f'{total_time2} secs, dt {time_step2}'])
plt.title("Central Differencing")

plt.figure(3)
plt.plot(grid_locations, intial_values)
plt.plot(grid_locations, solution3_5)
plt.plot(grid_locations, solution3)
plt.legend(['Initial Values', f'{total_time3/2} secs, dt {time_step3}', f'{total_time3} secs, dt {time_step3}'])
plt.title("Downstream Differencing")

plt.show()




# Plotting
