#########
# Solve for various vector operations including gradient, laplacian, divergence, etc.
########
from typing import List
import numpy as np

def gradient(field:list[list], dx, dy) -> list[list]:
    '''This function takes a numpy array and uses upwwind spatial
    differencing to calculate the gradient in the x and y direction. The returned values are stacked
    such that the gradient of y is appened to the gradient in the x direction'''
    
    #Determine length of gradient vector
    nx = len(field)
    ny = len(field[0])

    gradient_vector = np.zeros((nx-1)*ny+nx*(ny-1))

    ## First compute X component of gradient
    index = 0
    # First ssolve pinned point
    #i = 0, j=0
    gradient_vector[index] = (field[1,0]      ) / dx
    index += 1
    
    # Solve rest of bottom row
    j = 0
    for i in range(1, len(field)-1):
        gradient_vector[index] = (field[i+1,j] - field[i,j]) / dx
        index += 1

    # Solve rest of grid above first row
    for i in range(len(field)-1):
        for j in range(1, len(field[0])):
            gradient_vector[index] = (field[i+1,j] - field[i,j]) / dx
            index += 1

    ## Next compute Y component of gradient
    # First solve pinned point
    #i = 0, j=0
    gradient_vector[index] = (field[0,1]      ) / dy
    index += 1
    
    # Solve rest of left colum
    i = 0
    for j in range(1, len(field[0])-1):
        gradient_vector[index] = (field[i,j+1] - field[i,j]) / dy
        index += 1

    # Solve rest of grid above first row
    for j in range(len(field[0])-1):
        for i in range(1, len(field)):
            gradient_vector[index] = (field[i,j+1] - field[i,j]) / dy
            index += 1
            
    return gradient_vector


def div(u_vel, v_vel, nx, ny):
    '''This function computes the 2D divergence of a given vector field
    using an upwind spatial discretization scheme'''
    pass
    