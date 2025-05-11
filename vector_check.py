#### Used to write wrapper functions to check the vector solvers
# These functions assume the vector fields used take the following forms:
# U = sin(m*pi*x/Lx)*sin(n*pi*y/Ly)
# V = sin(n*pi*x/Lx)*sin(m*pi*y/Ly)
# m and n will be given when calling the functions
# Pressure = 

### Imports ###
import numpy as np

def calc_exact_div(u_vel, v_vel, dx, dy, m, n):
    '''This functions computes the exact divergence at the locations of pressure for the 
    given conditions'''

    # Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
    x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
    y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

    # Presure grid locations
    X_pressures, Y_pressures = np.meshgrid(x_array, y_array, indexing='ij')

    pass

def check_divergenc():
    pass

