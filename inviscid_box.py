# Simulate Potential Flow in a Box with the top surface moving

import numpy as np
from matplotlib import pyplot as plt

num_points = 10

x_points = np.linspace(0,1,num_points)
y_points = np.linspace(0,1,num_points)

X,Y = np.meshgrid(x_points, y_points)
# X = np.ravel(X)
# Y = np.ravel(Y)

# Initialize velocity field
# vel_field = np.zeros(num_points**2)
u_field = np.zeros((num_points,num_points))
v_field = np.zeros((num_points,num_points))
pres_field = np.ones((num_points,num_points))


#Apply initial conditions such that the top surface is moving right
for i in range(len(X)):
    for j in range(len(X[i])):
        if Y[i,j] == 1:
            u_field[i,j] = 1

# Solve for the velocity at everytpoint
