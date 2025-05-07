############
# This script initializes global variables for the rest of the oslver to use
# ##########
import numpy as np

#Initilize all global variables
nx = 0
ny = 0
dx = 0
dy = 0
u_vel = []
v_vel = []
pressures = []
q_vel = []

def Initialize_meshes(x_max, y_max, number_x_points, number_y_points):
    
    # Setup basic mesh parameters and declar them as global variables
    # number of grid cells
    global nx
    global ny
    # mesh spacing
    global dx
    global dy
    # velocity / pressure arrays
    global u_vel
    global v_vel
    global pressures
    # Combined velocity vector
    global q_vel

    ## Assign values to each global variable
    # Mesh conditions
    nx = number_x_points
    ny = number_y_points

    dx = x_max / number_x_points
    dy = y_max / number_y_points

    #Array initilizations
    # Create as 2D arrays that get stacked into a single vector q
    u_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
    v_vel = np.zeros((nx*ny)) #Slighlty larger than needed but makes iteration easier
    pressures = np.zeros((nx, ny)) #Slgihtly larger, one pressure should be pinned



