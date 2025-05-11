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
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TextIO, Tuple

# Constants
c = 1
RHS_func = lambda dfdx: -c*dfdx

#Initializations
num_points = 100
lower = 0
upper = 1
dx = (upper - lower) / num_points
grid = np.linspace(lower, upper, num_points)
starting_vals = []
exact_vals = []
######## Initial conditions and Exact solutions Comment / Uncomment desired function type #######

#Top hat
# for point in grid:
#     if point >= .3 and point <= .7:
#         starting_vals.append(1)
#     else:
#         starting_vals.append(0)

# for point in grid:
#     if point >= .8 and point <= 1.2:
#         exact_vals.append(1)
#     else:
#         exact_vals.append(0)

# cos wave
for point in grid:
    starting_vals.append(math.cos(point*2*math.pi))

#Exact Solution
for point in grid:
    exact_vals.append(math.cos((point-(c/2))*2*math.pi))
# ########

def Forward_Euler(function, current_solution, dx, dt, total_time_steps, spatial_method):
    '''Computes a forward Euler solution with a step size of dt for the given number of
     time steps. Requiress an array of the current time solution and the spatial grid size'''
    # intialize t=0
    t = 1

    while t <= total_time_steps:
        #Create array to store intermediate solution
        new_solution = []
        # Iterate through each point in current solution
        for i in range(len(current_solution)):
            new_solution.append(current_solution[i] + dt*function(get_dfdx(dx, spatial_method, current_solution, i)))

        # Update solution
        current_solution = new_solution

        # Iterate time step
        t += 1

    # Returned solved for values
    return current_solution

def Backward_Euler(function, current_solution, dx, dt, total_time_steps, spatial_method):
    '''Computes a forward Euler solution with a step size of dt for the given number of
     time steps. Requiress an array of the current time solution and the spatial grid size'''
    # intialize t=0
    t = 1

    while t <= total_time_steps:
        #Create array to store intermediate solution
        new_solution = []

        #Initialize implicit solver conditions
        epsilon = 1e-12
        iteration = 1
        max_iters = 5000
        # If first iteration produce a guess using forward euler
        yj = Forward_Euler(function, current_solution, dx, dt, 1, spatial_method)

        # Otherwise use a predictor corrector method
        # Iterate through each point in current solution
        for i in range(len(current_solution)):
            while True:

                new_guess = current_solution[i] + dt*function(get_dfdx(dx, spatial_method, yj, i))
                residual = new_guess-yj[i]

                if abs(residual) < epsilon or iteration > max_iters:
                    if iteration > max_iters:
                        print(f'Warning: Large residual possible residual={residual} with {max_iters} iterations')
                    break
        
                #update iterator and yj guess
                yj[i] = new_guess
                iteration += 1


            new_solution.append(current_solution[i] + dt*function(get_dfdx(dx, spatial_method, yj, i)))

        # Update solution
        current_solution = new_solution

        # Iterate time step
        t += 1

    # Returned solved for values
    return current_solution


def Heuns(function, current_solution, dx, dt, total_time_steps, spatial_method):
    '''Computes a Heuns methhod solution with a step size of dt for the given number of
     time steps. Requiress an array of the current time solution and the spatial grid size'''
    # intialize t=0
    t = 1

    while t <= total_time_steps:
        #Create array to store intermediate solution
        g_n = []
        f_bar = []
        new_solution = []
        # Iterate through each point in current solution
        for i in range(len(current_solution)):
            #Compute first intermediate solution
            g_n.append(function(get_dfdx(dx, spatial_method, current_solution, i)))
            f_bar.append(current_solution[i] + dt*(g_n[i]))
        for j in range(len(current_solution)):
            #Compute final solution
            new_solution.append(current_solution[j] + dt/2*(g_n[j] + function(get_dfdx(dx, spatial_method, f_bar, j))))
            

        # Update solution
        current_solution = new_solution

        # Iterate time step
        t += 1

    # Returned solved for values
    return current_solution


def RK_4(function, current_solution, dx, dt, total_time_steps, spatial_method):
    '''Computes a Runge-Kutta 4th order  solution with a step size of dt for the given number of
     time steps. Requiress an array of the current time solution and the spatial grid size'''
    # intialize t=0
    t = 1

    while t <= total_time_steps:
        #Create arrays to store intermediate solutions
        k1 = []
        g1 = []
        k2 = []
        g2 = []
        k3 = []
        g3 = []
        k4 = []
        new_solution = []
        # Iterate through each point in current solution
        # for i in range(len(current_solution)):
        #     g1.append(function(get_dfdx(dx, spatial_method, current_solution, i)))
        #     k1.append(current_solution[i] + dt/2*g1[i])
        #     g2.append(function(get_dfdx(dx, spatial_method, k1, i)))
        #     k2.append(current_solution[i] + dt/2*g2[i])
        #     g3.append(function(get_dfdx(dx, spatial_method, k2, i)))
        #     k3.append(current_solution[i] + dt*g2[i])
        #     new_solution.append(current_solution[i] + dt/6*(g))


        for i in range(len(current_solution)):
            # First step
            k1.append(dt*(function(get_dfdx(dx, spatial_method, current_solution, i))))
            g1.append(current_solution[i]+.5*k1[i])
        for i in range(len(current_solution)):
            # First step
            k2.append(dt*(function(get_dfdx(dx, spatial_method, g1, i))))
            g2.append(current_solution[i]+.5*k2[i])
        for i in range(len(current_solution)):
            # Second Step
            k3.append(dt*(function(get_dfdx(dx, spatial_method, g2, i))))
            g3.append(current_solution[i] + k3[i])
        for i in range(len(current_solution)):
            # Third Step
            k4.append(dt*(function(get_dfdx(dx, spatial_method, g3, i))))
        for i in range(len(current_solution)):
            # Solve
            new_solution.append(current_solution[i] + (1/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]))
            

        # Update solution
        current_solution = new_solution

        # Iterate time step
        t += 1

    # Returned solved for values
    return current_solution

# Create a function that returns the upwind spatial derivative
def get_dfdx(delta_x, scheme:str, values, i):
    '''scheme sets if upwind, central, or downwind differencing is used currently applies periodic boundary'''
    
    # Grab neighbor values from values, apply periodic boundary if at ends
    if i == 0:
        f0 = values[-1]
        f1 = values[i]
        f2 = values[i+1]  
    elif i== len(values)-1:
        f2 = values[0]
        f0 = values[i-1]
        f1 = values[i]
    else:
        f0 = values[i-1]
        f1 = values[i]
        f2 = values[i+1]      

    if scheme.lower() == "upwind":
        dfdx = (-f0 + f1) / delta_x
    elif scheme.lower() == "central":
        dfdx = (-f0 + f2) / (2*delta_x)
    elif scheme.lower() == "downwind":
        dfdx = (-f1 + f2) / delta_x
    else:
        raise SyntaxError(f"Invalid Spatial Derivative Scheme: {scheme} is not recognized")
    return dfdx

def calculate_L2_error(exact_solution, estimate_solution):
    # Assumes solutions have same length
    error = 0
    for i in range(len(exact_solution)):
        error += (abs(exact_solution[i] - estimate_solution[i])) / abs(exact_solution[i])
    return error


## Forward Euler Upwind

FE_up = Forward_Euler(RHS_func, starting_vals, dx, dt=.001, total_time_steps=500, spatial_method='upwind')
FE_up_error = calculate_L2_error(exact_vals, FE_up)

RK_4_up = RK_4(RHS_func, starting_vals, dx, dt=.001, total_time_steps=500, spatial_method='upwind')
RK4_up_error = calculate_L2_error(exact_vals, RK_4_up)

Heuns_up = Heuns(RHS_func, starting_vals, dx, dt=.001, total_time_steps=500, spatial_method='upwind')
Heuns_up_error = calculate_L2_error(exact_vals, Heuns_up)

BE_up = Backward_Euler(RHS_func, starting_vals, dx, dt=.001, total_time_steps=500, spatial_method='upwind')
BE_up_error = calculate_L2_error(exact_vals, BE_up)

BE_central = Backward_Euler(RHS_func, starting_vals, dx, dt=.001, total_time_steps=500, spatial_method='central')
BE_central_error = calculate_L2_error(exact_vals, BE_central)

#FE_central = Forward_Euler(RHS_func, starting_vals, dx, dt=.001, total_time_steps=1000, spatial_method='central')

error_test = calculate_L2_error(exact_vals, exact_vals)
error_test2 = calculate_L2_error(exact_vals, starting_vals)
error_test3 = calculate_L2_error(RK_4_up, Heuns_up)
print(error_test)
print(error_test2)
print(f'Diff between Rk4 and Heuns {error_test3}')


plt.figure(1)
plt.plot(grid, starting_vals)
plt.plot(grid, FE_up)
plt.plot(grid, RK_4_up)
plt.plot(grid, Heuns_up,linestyle='dashed')
plt.plot(grid, BE_up)
plt.plot(grid, BE_central)
plt.plot(grid, exact_vals, '-k')
#
plt.legend(['Initial Values', f'FE_Upwind error:{FE_up_error}',f'RK_4 error:{RK4_up_error}', f'Heuns error:{Heuns_up_error}', f'BE_Upwind error:{BE_up_error}', f'BE_Central error:{BE_central_error}', 'Exact Solution'])

# plt.figure(2)
# plt.plot(grid, FE_central)



##### Plot different solvers vs CFL number ######
total_time = .2
cfl_high = 0
cfl_low = -3
num_points = 10

cfl_numbers = np.logspace(cfl_low, cfl_high, num_points)
cfl_numbers = cfl_numbers[::-1]
error = []

exact_vals = []

#Exact Solution
for point in grid:
    exact_vals.append(math.cos((point-(total_time/c))*2*math.pi))

for cfl in cfl_numbers:
    dt = cfl*dx/c
    print(dt)
    timesteps = total_time/dt
    print(timesteps)
    solution = (RK_4(RHS_func, starting_vals, dx, dt, timesteps, spatial_method='upwind'))
    error.append(calculate_L2_error(exact_vals, solution))

plt.figure()
plt.loglog(cfl_numbers, error)
plt.xlabel('CFL Number')
plt.ylabel('L2 Error')


plt.figure()
plt.plot(grid, solution)
plt.plot(grid, exact_vals)
plt.legend(['Numeric', 'Exact'])

plt.show()