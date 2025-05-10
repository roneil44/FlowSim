#########
# Solve for various vector operations including gradient, laplacian, divergence, etc.
########
from typing import List, Tuple
import numpy as np


def gradient(field:list[list], dx:float, dy:float) -> list[list]:
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


def div(u_vel:list[list], v_vel:list[list], dx:float, dy:float) -> list:
    '''This function computes the 2D divergence of a given vector field
    using an midpoint spatial discretization scheme and returns a single
    linearized vector of length nq = (nx-1)*ny + nx*(ny-1)'''

    nx = len(u_vel)
    ny = len(u_vel[0])

    #nq = (nx-1)*ny + nx*(ny-1)
    n_p = nx*ny-1
    div_list = np.zeros(n_p)

    # Create lambda function to get index for given i ,j, assumes pinned pressure in 0,0
    xp = lambda i,j: i+j*nx - 1

    # We need to separate out the boundary conditions
    # Start along bottom excluding bottom corners
    j = 0
    for i in range(1, nx-1):
        div_list[xp(i,j)] = (u_vel[i+1,j] - u_vel[i,j]) / dx + (v_vel[i, j+1]       )/dy # -bottom_wall[1]/dy
    
    #Bottom right corner
    i=nx-1
    j=0
    div_list[xp(i,j)] = (      - u_vel[i,j]) / dx + (v_vel[i, j+1]        )/dy # right_wall[0]/dx + -bottom_wall[1]/dy
  
    #Left wall
    i = 0
    for j in range(1, ny-1):
        div_list[xp(i,j)] = (u_vel[i+1,j]      ) / dx + (v_vel[i, j+1] - v_vel[i, j])/dy # -left_wall[0]/dx

    #Right Wall
    i = nx-1
    for j in range(1, ny-1):
        div_list[xp(i,j)] = (      - u_vel[i,j]) / dx + (v_vel[i, j+1] - v_vel[i, j])/dy # right_wall[0]/dx

    #Top Wall
    j = ny-1
    for i in range(1, nx-1):
        div_list[xp(i,j)] = (u_vel[i+1,j] - u_vel[i,j]) / dx + (      - v_vel[i, j])/dy # top_wall[1]/dy

    #Top Left Corner
    i=0
    j=ny-1
    div_list[xp(i,j)] = (u_vel[i+1,j]       ) / dx + (      - v_vel[i, j])/dy #  -left_wall[0]/dx + top_wall[1]/dy

    #Top Right Corner
    i=nx-1
    j=ny-1
    div_list[xp(i,j)] = (      - u_vel[i,j]) / dx + (      - v_vel[i, j])/dy # right_wall[0]/dx + top_wall[1]/dy

    # All middle points
    for i in range(1,nx-1):
        for j in range(1, ny-1):
            div_list[xp(i,j)] = (u_vel[i+1,j] - u_vel[i,j]) / dx + (v_vel[i, j+1] - v_vel[i, j])/dy

    return div_list

def bc_div(u_vel:list[list], v_vel:list[list], dx:float, dy:float, top_wall:tuple, left_wall:tuple, right_wall:tuple, bottom_wall:tuple) -> list:
    '''This function computes the 2D divergence along the boundary of a square with the prescribed
    boundary conditionss given in each _wall tuple of (u, v) retunrs a vector of length np 
    where all non solved for points are 0'''

    nx = len(u_vel)
    ny = len(u_vel[0])

    #nq = (nx-1)*ny + nx*(ny-1)
    n_p = nx*ny-1
    div_list = np.zeros(n_p)

    # Create lambda function to get index for given i ,j, assumes pinned pressure in 0,0
    xp = lambda i,j: i+j*nx - 1

    # Start along bottom excluding bottom corners
    j = 0
    for i in range(1, nx-1):
        div_list[xp(i,j)] = -bottom_wall[1]/dy
    
    #Bottom right corner
    i=nx-1
    j=0
    div_list[xp(i,j)] = right_wall[0]/dx + -bottom_wall[1]/dy
  
    #Left wall
    i = 0
    for j in range(1, ny-1):
        div_list[xp(i,j)] = -left_wall[0]/dx

    #Right Wall
    i = nx-1
    for j in range(1, ny-1):
        div_list[xp(i,j)] = right_wall[0]/dx

    #Top Wall
    j = ny-1
    for i in range(1, nx-1):
        div_list[xp(i,j)] = top_wall[1]/dy

    #Top Left Corner
    i=0
    j=ny-1
    div_list[xp(i,j)] = -left_wall[0]/dx + top_wall[1]/dy

    #Top Right Corner
    i=nx-1
    j=ny-1
    div_list[xp(i,j)] = right_wall[0]/dx + top_wall[1]/dy

    return div_list


def lap(u_vel:list[list], v_vel:list[list], dx:float, dy:float) -> list:
    '''This function takes the two 2D arrays for u and v velocity and calculates the laplacian of the 
    2D velocity components. It returns these components as a single list of length '''
    #Initializations
    nx = len(u_vel)
    ny = len(u_vel[0])
    nq = (nx-1)*ny + nx*(ny-1)
    lap_list = np.zeros(nq)

    # Create lambda function to get index for given i ,j, assumes pinned pressure in 0,0
    xu = lambda i,j: i+j*(nx-1) - 1
    xv = lambda i,j: i+(j-1)*(nx) + (nx-1)*ny


    #### Udirection ####

    # Solve for bottom row, extrapolate U 1/2 cell below each point
    # U[i,j-1] = -u_vel[i,j]+2*bottom_wall[0]
    j=0
    for i in range(2, nx-1):
        lap_list[xu(i,j)] = (u_vel[i-1,j]-2*u_vel[i,j]+u_vel[i+1,j])/dx**2 + (     -2*u_vel[i,j]+u_vel[i,j+1])/dy**2 # -u_vel[i,j]+2*bottom_wall[0]/dy**2
    
    # Bottom Right corner
    j=0
    i=nx-1
    lap_list[xu(i,j)] = (u_vel[i-1,j]-2*u_vel[i,j]     )/dx**2 + (-2*u_vel[i,j]+u_vel[i,j+1])/dy**2 #right_wall[0]/dx**2 =(-u_vel[i,j]+2*bottom_wall[0])/dy**2

    # Solve for bottom left
    j=0
    i=1
    lap_list[xu(i,j)] = (-2*u_vel[i,j]+u_vel[i+1,j]    )/dx**2 + (    -2*u_vel[i,j]+u_vel[i,j+1])/dy**2 #left_wall[0]/dx**2 + (-u_vel[i,j]+2*bottom_wall[0])/dy**2 

    # Solve for left side inside wall
    i=1
    for j in range(1,ny-1):
        lap_list[xu(i,j)] = (     -2*u_vel[i,j]+u_vel[i+1,j])/dx**2 + (u_vel[i,j-1]-2*u_vel[i,j]+u_vel[i,j+1])/dy**2 #left_wall[0]/dx**2
    
    # Solve for right side inside wall
    i=nx-1
    for j in range(1,ny-1):
        lap_list[xu(i,j)] = (u_vel[i-1,j]-2*u_vel[i,j]     )/dx**2 + (u_vel[i,j-1]-2*u_vel[i,j]+u_vel[i,j+1])/dy**2 #right_wall[0]/dx**2

    # Solve for top row, extrapolate U 1/2 cell above each point
    # U[i,j+1] = -u_vel[i,j]+2*top_wall[0]
    j=ny-1
    for i in range(2, nx-1):
        lap_list[xu(i,j)] = (u_vel[i-1,j]-2*u_vel[i,j]+u_vel[i+1,j])/dx**2 + (u_vel[i,j-1]-2*u_vel[i,j]     )/dy**2 #(-u_vel[i,j]+2*top_wall[0])/dy**2

    # Solve for top left
    i=1
    j=ny-1
    lap_list[xu(i,j)] = (     -2*u_vel[i,j]+u_vel[i+1,j])/dx**2 + (u_vel[i,j-1]-2*u_vel[i,j]     )/dy**2 # left_wall[0]/dx**2 + (-u_vel[i,j]+2*top_wall[0])/dy**2

    #Solve for top right
    i = nx-1
    j = ny-1
    lap_list[xu(i,j)] = (u_vel[i-1,j]-2*u_vel[i,j]     )/dx**2 + (u_vel[i,j-1]-2*u_vel[i,j]     )/dy**2 # right_wall[0]/dx**2 + (-u_vel[i,j]+2*top_wall[0])/dy**2

    # Solve for center grid points
    for i in range(2, nx-1):
        for j in range(1, ny-1):
            lap_list[xu(i,j)] = (u_vel[i-1,j]-2*u_vel[i,j]+u_vel[i+1, j])/dx**2 + (u_vel[i,j-1]-2*u_vel[i,j]+u_vel[i,j+1])/dy**2
    
    
    #### VDirection ####
    #Solve for bottom row,
    j=1
    for i in range(1, nx-1):
        lap_list[xv(i,j)] = (v_vel[i-1,j]-2*v_vel[i,j]+v_vel[i+1,j])/dx**2 + (     -2*v_vel[i,j]+v_vel[i,j+1])/dy**2 # -u_vel[i,j]+2*bottom_wall[0]/dy**2
    
    # Bottom Right corner
    j=1
    i=nx-1
    lap_list[xv(i,j)] = (v_vel[i-1,j]-2*v_vel[i,j]     )/dx**2 + (-2*v_vel[i,j]+v_vel[i,j+1])/dy**2 #right_wall[0]/dx**2 =(-u_vel[i,j]+2*bottom_wall[0])/dy**2

    # Solve for bottom left
    j=1
    i=0
    lap_list[xv(i,j)] = (-2*v_vel[i,j]+v_vel[i+1,j]    )/dx**2 + (    -2*v_vel[i,j]+v_vel[i,j+1])/dy**2 #left_wall[0]/dx**2 + (-u_vel[i,j]+2*bottom_wall[0])/dy**2 

    # Solve for left side inside wall
    i=0
    for j in range(2,ny-1):
        lap_list[xv(i,j)] = (     -2*v_vel[i,j]+v_vel[i+1,j])/dx**2 + (v_vel[i,j-1]-2*v_vel[i,j]+v_vel[i,j+1])/dy**2 #left_wall[0]/dx**2
    
    # Solve for right side inside wall
    i=nx-1
    for j in range(2,ny-1):
        lap_list[xv(i,j)] = (v_vel[i-1,j]-2*v_vel[i,j]     )/dx**2 + (v_vel[i,j-1]-2*v_vel[i,j]+v_vel[i,j+1])/dy**2 #right_wall[0]/dx**2

    # Solve for top row
    j=ny-1
    for i in range(1, nx-1):
        #print(f'i:{i}, j:{j}')
        lap_list[xv(i,j)] = (v_vel[i-1,j]-2*v_vel[i,j]+v_vel[i+1,j])/dx**2 + (v_vel[i,j-1]-2*v_vel[i,j]     )/dy**2 #(-u_vel[i,j]+2*top_wall[0])/dy**2

    # Solve for top left
    i=0
    j=ny-1
    lap_list[xv(i,j)] = (     -2*v_vel[i,j]+v_vel[i+1,j])/dx**2 + (v_vel[i,j-1]-2*v_vel[i,j]     )/dy**2 # left_wall[0]/dx**2 + (-u_vel[i,j]+2*top_wall[0])/dy**2

    #Solve for top right
    i = nx-1
    j = ny-1
    lap_list[xv(i,j)] = (v_vel[i-1,j]-2*v_vel[i,j]     )/dx**2 + (v_vel[i,j-1]-2*v_vel[i,j]     )/dy**2 # right_wall[0]/dx**2 + (-u_vel[i,j]+2*top_wall[0])/dy**2

    # Solve for center grid points
    for i in range(1, nx-1):
        for j in range(2, ny-1):
            lap_list[xv(i,j)] = (v_vel[i-1,j]-2*v_vel[i,j]+v_vel[i+1, j])/dx**2 + (v_vel[i,j-1]-2*v_vel[i,j]+v_vel[i,j+1])/dy**2
    
    
    return lap_list


def bc_lap(u_vel:list[list], v_vel:list[list], dx:float, dy:float, top_wall:tuple, left_wall:tuple, right_wall:tuple, bottom_wall:tuple) -> list:
    '''This function takes the two 2D arrays for u and v velocity and calculates the Boundary conditions laplacian of the 
    2D velocity components. It returns these components as a single list of length nq'''
    #Initializations
    nx = len(u_vel)
    ny = len(u_vel[0])
    nq = (nx-1)*ny + nx*(ny-1)
    lap_list = np.zeros(nq)

    # Create lambda function to get index for given i ,j, assumes pinned pressure in 0,0
    xu = lambda i,j: i+j*(nx-1) - 1
    xv = lambda i,j: i+(j-1)*(nx) + (nx-1)*ny

    #### Udirection ####

    # Solve for bottom row, extrapolate U 1/2 cell below each point
    # U[i,j-1] = -u_vel[i,j]+2*bottom_wall[0]
    j=0
    for i in range(2, nx-1):
        lap_list[xu(i,j)] = (-u_vel[i,j]+2*bottom_wall[0])/dy**2
    
    # Bottom Right corner
    j=0
    i=nx-1
    lap_list[xu(i,j)] = right_wall[0]/dx**2 + (-u_vel[i,j]+2*bottom_wall[0])/dy**2

    # Solve for bottom left
    j=0
    i=1
    lap_list[xu(i,j)] = left_wall[0]/dx**2 + (-u_vel[i,j]+2*bottom_wall[0])/dy**2 

    # Solve for left side inside wall
    i=1
    for j in range(1,ny-1):
        lap_list[xu(i,j)] = left_wall[0]/dx**2
    
    # Solve for right side inside wall
    i=nx-1
    for j in range(1,ny-1):
        lap_list[xu(i,j)] = right_wall[0]/dx**2

    # Solve for top row, extrapolate U 1/2 cell above each point
    # U[i,j+1] = -u_vel[i,j]+2*top_wall[0]
    j=ny-1
    for i in range(2, nx-1):
        lap_list[xu(i,j)] = (-u_vel[i,j]+2*top_wall[0])/dy**2

    # Solve for top left
    i=1
    j=ny-1
    lap_list[xu(i,j)] = left_wall[0]/dx**2 + (-u_vel[i,j]+2*top_wall[0])/dy**2

    #Solve for top right
    i = nx-1
    j = ny-1
    lap_list[xu(i,j)] = right_wall[0]/dx**2 + (-u_vel[i,j]+2*top_wall[0])/dy**2
    
    #### VDirection ####
    #Solve for bottom row,
    j=1
    for i in range(1, nx-1):
        lap_list[xv(i,j)] = bottom_wall[1]/dy**2
    
    # Bottom Right corner
    j=1
    i=nx-1
    lap_list[xv(i,j)] = bottom_wall[1]/dy**2 + (-v_vel[i,j]+2*right_wall[1])/dx**2

    # Solve for bottom left
    j=1
    i=0
    lap_list[xv(i,j)] = bottom_wall[1]/dy**2 + (-v_vel[i,j]+2*left_wall[1])/dx**2 

    # Solve for left side inside wall
    i=0
    for j in range(2,ny-1):
        lap_list[xv(i,j)] = (-v_vel[i,j]+2*left_wall[1])/dx**2
    
    # Solve for right side inside wall
    i=nx-1
    for j in range(2,ny-1):
        lap_list[xv(i,j)] = (-v_vel[i,j]+2*right_wall[1])/dx**2

    # Solve for top row
    j=ny-1
    for i in range(1, nx-1):
        #print(f'i:{i}, j:{j}')
        lap_list[xv(i,j)] = top_wall[1]/dy**2

    # Solve for top left
    i=0
    j=ny-1
    lap_list[xv(i,j)] = top_wall[1]/dy**2 + (-v_vel[i,j]+2*left_wall[1])/dx**2

    #Solve for top right
    i = nx-1
    j = ny-1
    lap_list[xv(i,j)] = top_wall[1]/dy**2 + (-v_vel[i,j]+2*right_wall[1])/dx**2
    
    
    return lap_list


def advect(u_vel:list[list], v_vel:list[list], dx:float, dy:float, top_wall:tuple, left_wall:tuple, right_wall:tuple, bottom_wall:tuple) -> list:
    '''This function solves the nonlinear advection discretization used in the 2D
    incompressible Navier Stokes'''
    
    #Initializations
    nx = len(u_vel)
    ny = len(u_vel[0])
    nq = (nx-1)*ny + nx*(ny-1)
    adv_list = np.zeros(nq)

    # Create lambda function to get index for given i ,j, assumes pinned pressure in 0,0
    xu = lambda i,j: i+j*(nx-1) - 1
    xv = lambda i,j: i+(j-1)*(nx) + (nx-1)*ny

    #### Nx - Direction ######
    
    
    # Central points for Nx
    for i in range(2, nx-1):
        for j in range(1, ny-1):
            u_north = (u_vel[i, j+1] + u_vel[i,j]) / 2
            u_south = (u_vel[i, j-1] + u_vel[i,j]) / 2
            u_west = (u_vel[i-1, j] + u_vel[i,j]) / 2
            u_east = (u_vel[i+1, j] + u_vel[i,j]) / 2
            v_north = (v_vel[i, j+1] + v_vel[i,j]) / 2
            v_south = (v_vel[i, j-1] + v_vel[i,j]) / 2

            adv_list[xu(i,j)] = (u_east**2 - u_west**2)/dx + (u_north*v_north - u_south*v_south)/dy






    ##### Ny - Direction #######
    # Central points for Nx
    for i in range(1, nx-1):
        for j in range(2, ny-1):
            v_north = (v_vel[i, j+1] + v_vel[i,j]) / 2
            v_south = (v_vel[i, j-1] + v_vel[i,j]) / 2
            v_west = (v_vel[i-1, j] + v_vel[i,j]) / 2
            v_east = (v_vel[i+1, j] + v_vel[i,j]) / 2
            u_west = (u_vel[i, j-1] + u_vel[i,j]) / 2
            u_east = (u_vel[i+1, j-1] + u_vel[i+1,j]) / 2

            adv_list[xv(i,j)] = (v_north**2 - v_south**2)/dy + (v_east*u_east - v_west*u_west)/dx



            
            

    




